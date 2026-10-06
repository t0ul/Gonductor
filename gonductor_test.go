package gonductor_test

import (
	"context"
	"errors"
	"testing"

	"github.com/t0ul/gonductor"
)

// state mirrors the CaMeL loop shape enough to exercise the engine.
type state struct {
	steps   []string
	plan    string
	halted  bool
	counter int
}

func newCtx() *gonductor.Context {
	return &gonductor.Context{Context: context.Background(), TraceID: "t1"}
}

func TestLinearRunToEnd(t *testing.T) {
	record := func(name string) gonductor.Node[state] {
		return func(_ *gonductor.Context, s state) (state, error) {
			s.steps = append(s.steps, name)
			return s, nil
		}
	}
	g := gonductor.New[state]().
		AddNode("planner", record("planner")).
		AddNode("executor", record("executor")).
		AddNode("coder", record("coder")).
		SetEntry("planner").
		AddEdge("planner", "executor").
		AddEdge("executor", "coder").
		AddEdge("coder", gonductor.END)

	c, err := g.Compile()
	if err != nil {
		t.Fatal(err)
	}
	final, err := c.Invoke(newCtx(), state{})
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"planner", "executor", "coder"}
	if len(final.steps) != 3 || final.steps[0] != want[0] || final.steps[2] != want[2] {
		t.Fatalf("path = %v, want %v", final.steps, want)
	}
}

func TestConditionalEdgeRoutes(t *testing.T) {
	g := gonductor.New[state]().
		AddNode("planner", func(_ *gonductor.Context, s state) (state, error) {
			s.plan = "EXECUTION_HALTED"
			return s, nil
		}).
		AddNode("executor", func(_ *gonductor.Context, s state) (state, error) {
			s.steps = append(s.steps, "executor")
			return s, nil
		}).
		AddNode("halt", func(_ *gonductor.Context, s state) (state, error) {
			s.halted = true
			return s, nil
		}).
		SetEntry("planner").
		AddConditionalEdge("planner", func(s state) string {
			if s.plan == "EXECUTION_HALTED" {
				return "halt"
			}
			return "executor"
		}).
		AddEdge("executor", gonductor.END).
		AddEdge("halt", gonductor.END)

	c, _ := g.Compile()
	final, err := c.Invoke(newCtx(), state{})
	if err != nil {
		t.Fatal(err)
	}
	if !final.halted || len(final.steps) != 0 {
		t.Fatalf("expected halt branch, got halted=%v steps=%v", final.halted, final.steps)
	}
}

func TestHardCircuitBreaker(t *testing.T) {
	// planner loops to itself forever; MaxSteps must stop it.
	g := gonductor.New[state]().
		AddNode("planner", func(_ *gonductor.Context, s state) (state, error) {
			s.counter++
			return s, nil
		}).
		SetEntry("planner").
		AddEdge("planner", "planner").
		MaxSteps(5)

	c, _ := g.Compile()
	final, err := c.Invoke(newCtx(), state{})
	if !errors.Is(err, gonductor.ErrCircuitBreaker) {
		t.Fatalf("expected ErrCircuitBreaker, got %v", err)
	}
	if final.counter != 5 {
		t.Fatalf("expected exactly 5 steps before trip, got %d", final.counter)
	}
}

func TestAppLevelGracefulHalt(t *testing.T) {
	// A node reads ctx.Step and halts itself before the hard cap.
	g := gonductor.New[state]().
		AddNode("planner", func(ctx *gonductor.Context, s state) (state, error) {
			if ctx.Step >= 3 {
				s.plan = "EXECUTION_HALTED"
			}
			s.counter++
			return s, nil
		}).
		SetEntry("planner").
		AddConditionalEdge("planner", func(s state) string {
			if s.plan == "EXECUTION_HALTED" {
				return gonductor.END
			}
			return "planner"
		}).
		MaxSteps(100)

	c, _ := g.Compile()
	final, err := c.Invoke(newCtx(), state{})
	if err != nil {
		t.Fatalf("graceful halt should not error, got %v", err)
	}
	if final.counter != 3 {
		t.Fatalf("expected graceful halt at step 3, got %d", final.counter)
	}
}

func TestStepHookFires(t *testing.T) {
	var seen []string
	g := gonductor.New[state]().
		AddNode("a", func(_ *gonductor.Context, s state) (state, error) { return s, nil }).
		AddNode("b", func(_ *gonductor.Context, s state) (state, error) { return s, nil }).
		SetEntry("a").
		AddEdge("a", "b").
		AddEdge("b", gonductor.END).
		OnStep(func(ctx *gonductor.Context, node string, _ state) {
			seen = append(seen, node)
			if ctx.Step == 0 {
				t.Error("step should be incremented before hook")
			}
		})
	c, _ := g.Compile()
	if _, err := c.Invoke(newCtx(), state{}); err != nil {
		t.Fatal(err)
	}
	if len(seen) != 2 || seen[0] != "a" || seen[1] != "b" {
		t.Fatalf("hook saw %v, want [a b]", seen)
	}
}

func TestNodeErrorAborts(t *testing.T) {
	boom := errors.New("boom")
	g := gonductor.New[state]().
		AddNode("x", func(_ *gonductor.Context, s state) (state, error) { return s, boom }).
		SetEntry("x").
		AddEdge("x", gonductor.END)
	c, _ := g.Compile()
	if _, err := c.Invoke(newCtx(), state{}); !errors.Is(err, boom) {
		t.Fatalf("expected wrapped boom, got %v", err)
	}
}

func TestContextCancellationStops(t *testing.T) {
	ctxStd, cancel := context.WithCancel(context.Background())
	cancel()
	gctx := &gonductor.Context{Context: ctxStd, TraceID: "t"}
	g := gonductor.New[state]().
		AddNode("x", func(_ *gonductor.Context, s state) (state, error) {
			t.Error("node should not run after cancel")
			return s, nil
		}).
		SetEntry("x").
		AddEdge("x", gonductor.END)
	c, _ := g.Compile()
	if _, err := c.Invoke(gctx, state{}); !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", err)
	}
}

func TestCompileRejectsBadGraph(t *testing.T) {
	cases := map[string]*gonductor.Graph[state]{
		"no entry":       gonductor.New[state]().AddNode("a", nil),
		"entry missing":  gonductor.New[state]().AddNode("a", nil).SetEntry("z"),
		"edge to nowhere": gonductor.New[state]().AddNode("a", nil).SetEntry("a").AddEdge("a", "ghost"),
	}
	for name, g := range cases {
		if _, err := g.Compile(); err == nil {
			t.Errorf("%s: expected compile error, got nil", name)
		}
	}
}
