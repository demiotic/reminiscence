---
title: Multi-Agent Systems
description: Caching multi-agent workflows and collaborative AI systems
---

This guide shows how to cache multi-agent systems where multiple AI agents collaborate.

## Basic Multi-Agent System

```python
from reminiscence import Reminiscence
from openai import OpenAI

cache = Reminiscence()
client = OpenAI()

# Different agents with separate caches
@cache.cached(query="task", context=["agent_id", "model"])
def agent_execute(task: str, agent_id: str, model: str = "gpt-4"):
    """Execute task with agent-specific caching"""
    system_prompts = {
        "researcher": "You are a research assistant. Provide detailed, factual information.",
        "writer": "You are a creative writer. Write engaging, narrative content.",
        "analyst": "You are a data analyst. Provide analytical insights.",
        "critic": "You are a critical reviewer. Evaluate and provide feedback."
    }

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompts[agent_id]},
            {"role": "user", "content": task}
        ]
    )

    return response.choices[0].message.content

# Each agent has isolated cache
research = agent_execute("Explain quantum computing", agent_id="researcher")
writing = agent_execute("Explain quantum computing", agent_id="writer")  # Different cache
analysis = agent_execute("Analyze trends", agent_id="analyst")
```

## Sequential Agent Pipeline

Agents work in sequence, each caching independently:

```python
@cache.cached(query="task", context=["agent_id", "previous_output_hash"])
def agent_with_context(
    task: str,
    agent_id: str,
    previous_output: str = ""
):
    """Agent execution with previous agent's output"""
    system_prompts = {
        "planner": "Break down the task into steps",
        "executor": "Execute the planned steps",
        "reviewer": "Review and improve the execution"
    }

    if previous_output:
        full_prompt = f"Previous output:\n{previous_output}\n\nYour task: {task}"
    else:
        full_prompt = task

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompts[agent_id]},
            {"role": "user", "content": full_prompt}
        ]
    )

    return response.choices[0].message.content

def sequential_agents(task: str):
    """Sequential agent pipeline with caching"""
    # Step 1: Planner
    plan = agent_with_context(task, agent_id="planner")

    # Step 2: Executor (uses plan)
    execution = agent_with_context(
        "Execute this plan",
        agent_id="executor",
        previous_output=plan
    )

    # Step 3: Reviewer (reviews execution)
    review = agent_with_context(
        "Review and improve",
        agent_id="reviewer",
        previous_output=execution
    )

    return {
        "plan": plan,
        "execution": execution,
        "review": review
    }

# Entire pipeline cached at each step
result = sequential_agents("Create a blog post about AI")
```

## Parallel Agent Execution

Multiple agents work in parallel:

```python
from concurrent.futures import ThreadPoolExecutor

@cache.cached(query="task", context=["agent_id", "specialization"])
def specialized_agent(
    task: str,
    agent_id: str,
    specialization: str
):
    """Specialized agent execution"""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": f"You are a {specialization} specialist."
            },
            {"role": "user", "content": task}
        ]
    )

    return {
        "agent": agent_id,
        "specialization": specialization,
        "response": response.choices[0].message.content
    }

def parallel_agents(task: str):
    """Execute agents in parallel with caching"""
    agents = [
        {"agent_id": "tech_expert", "specialization": "technology"},
        {"agent_id": "business_expert", "specialization": "business"},
        {"agent_id": "legal_expert", "specialization": "legal"},
        {"agent_id": "marketing_expert", "specialization": "marketing"}
    ]

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(
                specialized_agent,
                task,
                agent["agent_id"],
                agent["specialization"]
            )
            for agent in agents
        ]

        results = [f.result() for f in futures]

    return results

# All agents execute in parallel, each with caching
results = parallel_agents("Analyze the impact of AI on society")
```

## Agent Collaboration

Agents collaborate and vote:

```python
@cache.cached(
    query="task",
    context=["consensus_agents", "model"]
)
def consensus_agents(
    task: str,
    agent_roles: list[str],
    model: str = "gpt-4"
):
    """Multiple agents reach consensus"""
    # Each agent generates response
    responses = []

    for role in agent_roles:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": f"You are a {role}."},
                {"role": "user", "content": task}
            ]
        )
        responses.append({
            "role": role,
            "response": response.choices[0].message.content
        })

    # Synthesizer agent combines responses
    combined = "\n\n".join([
        f"{r['role']}: {r['response']}"
        for r in responses
    ])

    synthesis = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Synthesize these perspectives into a consensus."
            },
            {"role": "user", "content": combined}
        ]
    )

    return {
        "individual_responses": responses,
        "consensus": synthesis.choices[0].message.content
    }

# Entire consensus process cached
result = consensus_agents(
    "Should we invest in AI?",
    agent_roles=["investor", "technologist", "ethicist"]
)
```

## Agent with Memory

Agent with persistent memory across tasks:

```python
class AgentWithMemory:
    def __init__(self, agent_id: str, cache: Reminiscence):
        self.agent_id = agent_id
        self.cache = cache
        self.memory = []

    @cache.cached(
        query="task",
        context=["agent_id", "memory_hash"]
    )
    def execute(self, task: str):
        """Execute with memory context"""
        # Include memory in context
        memory_context = "\n".join([
            f"[{i}] {mem}" for i, mem in enumerate(self.memory[-5:])  # Last 5
        ])

        full_prompt = f"Previous interactions:\n{memory_context}\n\nCurrent task: {task}"

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": f"You are agent {self.agent_id}"},
                {"role": "user", "content": full_prompt}
            ]
        )

        result = response.choices[0].message.content

        # Update memory
        self.memory.append(f"Task: {task}, Response: {result[:100]}...")

        return result

# Agent remembers across tasks
agent = AgentWithMemory("assistant_1", cache)

response1 = agent.execute("What is 2+2?")
response2 = agent.execute("What was my previous question?")  # Uses memory
```

## Task Decomposition

Agent breaks down and caches subtasks:

```python
@cache.cached(query="task", context=["decomposer_agent"])
def decompose_task(task: str):
    """Decompose task into subtasks"""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "Break down tasks into specific subtasks. Return as JSON array."
            },
            {"role": "user", "content": task}
        ],
        response_format={"type": "json_object"}
    )

    return json.loads(response.choices[0].message.content)

@cache.cached(query="subtask", context=["executor_agent"])
def execute_subtask(subtask: str):
    """Execute individual subtask"""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Execute this specific subtask."},
            {"role": "user", "content": subtask}
        ]
    )

    return response.choices[0].message.content

def hierarchical_execution(task: str):
    """Decompose and execute with caching at each level"""
    # Decompose (cached)
    decomposition = decompose_task(task)
    subtasks = decomposition["subtasks"]

    # Execute subtasks (each cached independently)
    results = []
    for subtask in subtasks:
        result = execute_subtask(subtask)
        results.append({
            "subtask": subtask,
            "result": result
        })

    # Combine results
    combined = "\n\n".join([
        f"Subtask: {r['subtask']}\nResult: {r['result']}"
        for r in results
    ])

    # Final synthesis (cached)
    synthesis = synthesize_results(combined, task)

    return {
        "decomposition": subtasks,
        "subtask_results": results,
        "final_result": synthesis
    }

@cache.cached(query="combined", context=["synthesizer", "original_task_hash"])
def synthesize_results(combined: str, original_task: str):
    """Synthesize subtask results"""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "Synthesize subtask results for the original task."
            },
            {
                "role": "user",
                "content": f"Original task: {original_task}\n\nResults:\n{combined}"
            }
        ]
    )

    return response.choices[0].message.content

# Complete hierarchical execution cached
result = hierarchical_execution("Plan and execute a marketing campaign")
```

## Multi-Agent Debate

Agents debate and refine responses:

```python
@cache.cached(
    query="topic",
    context=["debate_agents", "rounds"]
)
def agent_debate(
    topic: str,
    agent_perspectives: list[str],
    rounds: int = 3
):
    """Multi-agent debate with caching"""
    debate_history = []

    for round_num in range(rounds):
        round_responses = []

        for perspective in agent_perspectives:
            # Agent sees previous round
            context = "\n\n".join([
                f"Round {i+1}:\n{entry}"
                for i, entry in enumerate(debate_history)
            ])

            prompt = f"Debate topic: {topic}\n\nPrevious discussion:\n{context}\n\nYour perspective ({perspective}):"

            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": f"Argue from the {perspective} perspective."
                    },
                    {"role": "user", "content": prompt}
                ]
            )

            round_responses.append({
                "perspective": perspective,
                "response": response.choices[0].message.content
            })

        debate_history.append("\n".join([
            f"{r['perspective']}: {r['response']}"
            for r in round_responses
        ]))

    # Final consensus
    all_arguments = "\n\n".join(debate_history)

    consensus = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "Synthesize the debate into a balanced conclusion."
            },
            {"role": "user", "content": all_arguments}
        ]
    )

    return {
        "debate_rounds": debate_history,
        "consensus": consensus.choices[0].message.content
    }

# Entire debate cached
result = agent_debate(
    "Should AI be regulated?",
    agent_perspectives=["proponent", "skeptic", "pragmatist"],
    rounds=3
)
```

## Agent Routing

Route tasks to appropriate agents:

```python
@cache.cached(query="task", context=["router_model"])
def route_task(task: str):
    """Determine which agent should handle task"""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": """Classify the task type. Return JSON:
                {"agent": "research|analysis|creative|technical",
                 "confidence": 0.0-1.0}"""
            },
            {"role": "user", "content": task}
        ],
        response_format={"type": "json_object"}
    )

    return json.loads(response.choices[0].message.content)

@cache.cached(query="task", context=["assigned_agent"])
def execute_routed_task(task: str, agent_type: str):
    """Execute with assigned agent"""
    agent_configs = {
        "research": "Provide detailed research and citations",
        "analysis": "Provide data-driven analysis",
        "creative": "Generate creative content",
        "technical": "Provide technical implementation details"
    }

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": agent_configs[agent_type]},
            {"role": "user", "content": task}
        ]
    )

    return response.choices[0].message.content

def smart_routing(task: str):
    """Route and execute with caching"""
    # Route (cached)
    routing = route_task(task)

    # Execute with appropriate agent (cached)
    result = execute_routed_task(task, routing["agent"])

    return {
        "routed_to": routing["agent"],
        "confidence": routing["confidence"],
        "result": result
    }

# Both routing and execution cached
result1 = smart_routing("Analyze customer churn data")  # → analysis agent
result2 = smart_routing("Write a blog post")  # → creative agent
```

## Agent Feedback Loop

Agents iterate with feedback:

```python
@cache.cached(
    query="task",
    context=["generator_agent", "iteration"]
)
def generator_agent(task: str, iteration: int = 0):
    """Generate content"""
    prompt = f"Generate content for: {task}"

    if iteration > 0:
        prompt += f" (Iteration {iteration} - refine based on feedback)"

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

@cache.cached(query="content_hash", context=["critic_agent"])
def critic_agent(content: str):
    """Provide feedback"""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "Provide constructive feedback. Return JSON with 'score' (0-10) and 'feedback'."
            },
            {"role": "user", "content": content}
        ],
        response_format={"type": "json_object"}
    )

    return json.loads(response.choices[0].message.content)

def feedback_loop(task: str, max_iterations: int = 3, target_score: float = 8.0):
    """Iterate until quality threshold met"""
    iteration = 0

    while iteration < max_iterations:
        # Generate (cached per iteration)
        content = generator_agent(task, iteration)

        # Critique (cached)
        feedback = critic_agent(content)

        if feedback["score"] >= target_score:
            return {
                "content": content,
                "iterations": iteration + 1,
                "final_score": feedback["score"]
            }

        iteration += 1

    return {
        "content": content,
        "iterations": max_iterations,
        "final_score": feedback["score"],
        "note": "Max iterations reached"
    }

# Feedback loop with caching at each step
result = feedback_loop("Write a product description for AI software")
```

## Multi-Agent Workflow Orchestration

Complete workflow with multiple agent types:

```python
class AgentOrchestrator:
    def __init__(self, cache: Reminiscence):
        self.cache = cache

    @cache.cached(query="request", context=["workflow_stage", "agent_roles"])
    def execute_stage(
        self,
        request: str,
        stage: str,
        agent_roles: list[str]
    ):
        """Execute workflow stage with multiple agents"""
        stage_prompts = {
            "planning": "Create a detailed plan",
            "execution": "Execute the plan",
            "review": "Review and provide feedback",
            "finalization": "Finalize the output"
        }

        results = []
        for role in agent_roles:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": f"You are a {role}."},
                    {
                        "role": "user",
                        "content": f"{stage_prompts[stage]}: {request}"
                    }
                ]
            )

            results.append({
                "role": role,
                "output": response.choices[0].message.content
            })

        return results

    def orchestrate(self, request: str):
        """Full workflow orchestration"""
        # Stage 1: Planning
        planning = self.execute_stage(
            request,
            stage="planning",
            agent_roles=["strategist", "analyst"]
        )

        # Stage 2: Execution
        execution = self.execute_stage(
            request,
            stage="execution",
            agent_roles=["developer", "designer"]
        )

        # Stage 3: Review
        review = self.execute_stage(
            request,
            stage="review",
            agent_roles=["qa_specialist", "security_expert"]
        )

        # Stage 4: Finalization
        finalization = self.execute_stage(
            request,
            stage="finalization",
            agent_roles=["technical_writer", "product_manager"]
        )

        return {
            "planning": planning,
            "execution": execution,
            "review": review,
            "finalization": finalization
        }

# Complete workflow with stage-level caching
orchestrator = AgentOrchestrator(cache)
result = orchestrator.orchestrate("Build a user authentication system")
```

## Next Steps

- [LLM Applications](/examples/llm-apps/) - General LLM caching patterns
- [RAG Pipelines](/examples/rag/) - Caching RAG systems
- [Decorators](/guides/decorators/) - Advanced decorator patterns
- [Best Practices](/production/best-practices/) - Production deployment
