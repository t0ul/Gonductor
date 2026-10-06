// Package gonductor is a small, dependency-free state-graph engine for driving
// multi-step LLM agent loops — the Go replacement for the LangGraph StateGraph
// in the CaMeL control plane.
//
// A graph is a set of named nodes over a shared, caller-defined state value S.
// Each node transforms the state; edges (static or conditional) decide the next
// node; the run ends at END. The engine is deliberately minimal: it owns
// control flow, cancellation, and a hard step cap (the runaway-loop / sponge
// safety stop, OWASP LLM DoS). It takes no opinion on what a node does, so the
// same engine drives planner → approval → executor → coder or any other loop.
//
// Two layers of circuit breaking, by design (defense in depth):
//   - engine-level: MaxSteps is a hard cap; exceeding it fails the run with
//     ErrCircuitBreaker, so a buggy routing cycle cannot spin forever.
//   - app-level: the current step is exposed on *Context, so a node can route
//     to a graceful halt node of its own before the hard cap is hit.
package gonductor

import (
	"context"
	"errors"
	"fmt"
)

// END is the terminal pseudo-node. Routing to it ends the run.
const END = "__end__"

// ErrCircuitBreaker is returned when a run exceeds MaxSteps.
var ErrCircuitBreaker = errors.New("gonductor: circuit breaker tripped (max steps exceeded)")

// Context carries the run's correlation id, step counter, and the standard
// context.Context for cancellation and deadlines. It is passed to every node.
type Context struct {
	context.Context
	TraceID string
	// Step is the number of nodes entered so far in this run (1 on the first
	// node). Nodes may read it to implement their own graceful halt.
	Step int
}

// Node transforms the state for one step. Returning an error aborts the run.
type Node[S any] func(ctx *Context, state S) (S, error)

// Router picks the next node name from the post-node state. Returning END (or a
// node that does not exist, which Compile rejects) ends the run.
type Router[S any] func(state S) string

// StepHook observes each node entry (for audit/telemetry spans). It runs before
// the node, with the step count already incremented on ctx.
type StepHook[S any] func(ctx *Context, node string, state S)

// Graph is the mutable builder. Build it, then Compile for a runnable graph.
type Graph[S any] struct {
	nodes    map[string]Node[S]
	edges    map[string]string
	routers  map[string]Router[S]
	entry    string
	maxSteps int
	onStep   StepHook[S]
}

// New returns an empty graph.
func New[S any]() *Graph[S] {
	return &Graph[S]{
		nodes:   map[string]Node[S]{},
		edges:   map[string]string{},
		routers: map[string]Router[S]{},
	}
}

// AddNode registers a node by name (replacing any node of that name).
func (g *Graph[S]) AddNode(name string, n Node[S]) *Graph[S] {
	g.nodes[name] = n
	return g
}

// AddEdge adds a static transition from -> to. to may be END.
func (g *Graph[S]) AddEdge(from, to string) *Graph[S] {
	g.edges[from] = to
	return g
}

// AddConditionalEdge routes out of from by calling route(state). A conditional
// edge takes precedence over a static edge on the same node.
func (g *Graph[S]) AddConditionalEdge(from string, route Router[S]) *Graph[S] {
	g.routers[from] = route
	return g
}

// SetEntry names the first node.
func (g *Graph[S]) SetEntry(name string) *Graph[S] {
	g.entry = name
	return g
}

// MaxSteps sets the hard step cap (<= 0 means unlimited). This is the engine's
// runaway-loop safety stop; prefer a small positive value for agent loops.
func (g *Graph[S]) MaxSteps(n int) *Graph[S] {
	g.maxSteps = n
	return g
}

// OnStep registers a hook called on each node entry, for audit spans.
func (g *Graph[S]) OnStep(fn StepHook[S]) *Graph[S] {
	g.onStep = fn
	return g
}

// Compiled is a validated, runnable graph.
type Compiled[S any] struct{ g *Graph[S] }

// Compile validates the graph: the entry node must be set and exist, and every
// static edge target must be a known node or END. Conditional targets are
// checked at run time (they are data-dependent).
func (g *Graph[S]) Compile() (*Compiled[S], error) {
	if g.entry == "" {
		return nil, errors.New("gonductor: no entry node set")
	}
	if _, ok := g.nodes[g.entry]; !ok {
		return nil, fmt.Errorf("gonductor: entry node %q is not registered", g.entry)
	}
	for from, to := range g.edges {
		if _, ok := g.nodes[from]; !ok {
			return nil, fmt.Errorf("gonductor: edge from unknown node %q", from)
		}
		if to != END {
			if _, ok := g.nodes[to]; !ok {
				return nil, fmt.Errorf("gonductor: edge %q -> unknown node %q", from, to)
			}
		}
	}
	for from := range g.routers {
		if _, ok := g.nodes[from]; !ok {
			return nil, fmt.Errorf("gonductor: conditional edge from unknown node %q", from)
		}
	}
	return &Compiled[S]{g: g}, nil
}

// next resolves the node to run after cur. Conditional edge wins; then static
// edge; a node with neither routes to END.
func (g *Graph[S]) next(cur string, state S) string {
	if r, ok := g.routers[cur]; ok {
		return r(state)
	}
	if to, ok := g.edges[cur]; ok {
		return to
	}
	return END
}

// Invoke runs the graph from the entry node until END, an error, a tripped
// circuit breaker, or ctx cancellation. It returns the final state. ctx.TraceID
// should be set by the caller; ctx.Step is managed by the engine.
func (c *Compiled[S]) Invoke(ctx *Context, state S) (S, error) {
	g := c.g
	cur := g.entry
	for cur != END {
		if err := ctx.Err(); err != nil {
			return state, err
		}
		if g.maxSteps > 0 && ctx.Step >= g.maxSteps {
			return state, ErrCircuitBreaker
		}
		node, ok := g.nodes[cur]
		if !ok {
			return state, fmt.Errorf("gonductor: routed to unknown node %q", cur)
		}
		ctx.Step++
		if g.onStep != nil {
			g.onStep(ctx, cur, state)
		}
		ns, err := node(ctx, state)
		if err != nil {
			return state, fmt.Errorf("gonductor: node %q: %w", cur, err)
		}
		state = ns
		cur = g.next(cur, state)
	}
	return state, nil
}
