"""
SmolLM2 Agent with Neo4j Integration

Implements a ReAct-style agent loop where SmolLM2 decides actions
and executes them via Neo4j Python driver.
"""

import json
import re
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

import requests

from .neo4j_client import Neo4jClient, Neo4jResult


class AgentAction(Enum):
    """Types of agent actions."""
    CALL_TOOL = "call_tool"
    FINAL_ANSWER = "final_answer"
    THINKING = "thinking"
    ERROR = "error"


@dataclass
class AgentStep:
    """A single step in the agent's reasoning process."""
    timestamp: str
    thought: str
    action: AgentAction
    tool_name: Optional[str] = None
    tool_input: Optional[Dict] = None
    observation: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "thought": self.thought,
            "action": self.action.value,
            "tool_name": self.tool_name,
            "tool_input": self.tool_input,
            "observation": self.observation
        }


@dataclass
class AgentResult:
    """Result of an agent run."""
    success: bool
    answer: str
    steps: List[AgentStep] = field(default_factory=list)
    error: Optional[str] = None
    total_iterations: int = 0

    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "answer": self.answer,
            "steps": [s.to_dict() for s in self.steps],
            "error": self.error,
            "total_iterations": self.total_iterations
        }


class SmolLM2Agent:
    """
    ReAct Agent using SmolLM2 via Ollama with Neo4j tool access.

    This agent implements the ReAct (Reasoning + Acting) pattern:
    1. Receive a task
    2. Think about what to do
    3. Decide to call a tool or provide final answer
    4. If tool called, observe the result
    5. Repeat until task is complete

    Usage:
        client = Neo4jClient()
        client.connect()

        agent = SmolLM2Agent(neo4j_client=client, verbose=True)
        result = agent.run("What nodes exist in the database?")

        print(result.answer)
        for step in result.steps:
            print(f"{step.thought} -> {step.action}")
    """

    OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
    OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"

    def __init__(
        self,
        model: str = "hf.co/HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF:Q4_K_M",
        neo4j_client: Optional[Neo4jClient] = None,
        max_iterations: int = 10,
        verbose: bool = True,
        temperature: float = 0.1,
        timeout: int = 120,
        on_step: Optional[Callable[[AgentStep], None]] = None
    ):
        """
        Initialize SmolLM2 Agent.

        Args:
            model: Ollama model name
            neo4j_client: Connected Neo4jClient instance
            max_iterations: Maximum reasoning iterations
            verbose: Print debug information
            temperature: LLM temperature (lower = more deterministic)
            timeout: Request timeout in seconds
            on_step: Callback function called after each step
        """
        self.model = model
        self.neo4j_client = neo4j_client
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.temperature = temperature
        self.timeout = timeout
        self.on_step = on_step
        self.history: List[AgentStep] = []

    def _log(self, message: str):
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            print(f"[Agent] {message}")

    def _get_system_prompt(self, context: Optional[Dict] = None) -> str:
        """Generate the system prompt for the agent."""
        prompt = """You are a helpful data modeling assistant that helps users build and manage Neo4j knowledge graphs.

You have access to the following tools:

1. get_schema - Get the current schema of the Neo4j database (labels and relationship types)
   Parameters: none

2. run_cypher - Execute a read Cypher query
   Parameters: query (string, required)

3. write_cypher - Execute a write Cypher query to create/update data
   Parameters: query (string, required)

When you need to use a tool, respond EXACTLY in this format:
THOUGHT: [your reasoning about what to do]
ACTION: [tool_name]
ACTION_INPUT: [JSON object with parameters]

When you have completed the task and want to provide the final answer:
THOUGHT: [your final reasoning]
FINAL_ANSWER: [your complete answer to the user]

Important rules:
- Always start with THOUGHT
- Use get_schema first to understand the current database state
- Use valid Cypher syntax for queries
- For creating nodes, use MERGE to avoid duplicates
- Always include a key property (like id or name) for nodes
"""

        if context:
            prompt += f"\n\nContext provided by user:\n{json.dumps(context, indent=2)}"

        return prompt

    def _call_ollama_generate(self, prompt: str) -> str:
        """Call Ollama generate API."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": 1000
            }
        }

        self._log(f"Calling Ollama generate API...")
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"[Agent OLLAMA] === PROMPT START ===")
            print(prompt)
            print(f"[Agent OLLAMA] === PROMPT END === ({len(prompt)} chars)")
            print(f"{'='*80}")

        response = requests.post(
            self.OLLAMA_GENERATE_URL,
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        result = response.json().get("response", "")

        if self.verbose:
            print(f"[Agent OLLAMA] === RESPONSE START ===")
            print(result)
            print(f"[Agent OLLAMA] === RESPONSE END === ({len(result)} chars)")
            print(f"{'='*80}")

        return result

    def _parse_agent_response(self, response: str) -> tuple[AgentAction, str, Optional[str], Optional[Dict]]:
        """
        Parse the agent's response to extract action and parameters.

        Returns:
            (action_type, thought, tool_name, tool_input)
        """
        # Extract THOUGHT
        thought_match = re.search(r'THOUGHT:\s*(.+?)(?=ACTION:|FINAL_ANSWER:|$)', response, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else "Thinking..."

        # Check for FINAL_ANSWER
        final_match = re.search(r'FINAL_ANSWER:\s*(.+?)$', response, re.DOTALL)
        if final_match:
            return AgentAction.FINAL_ANSWER, thought, None, {"answer": final_match.group(1).strip()}

        # Check for ACTION
        action_match = re.search(r'ACTION:\s*([\w_]+)', response)
        if action_match:
            tool_name = action_match.group(1).strip()

            # Extract ACTION_INPUT
            input_match = re.search(r'ACTION_INPUT:\s*(\{.+?\}|.+?)(?=\n|$)', response, re.DOTALL)
            tool_input = {}
            if input_match:
                try:
                    input_str = input_match.group(1).strip()
                    tool_input = json.loads(input_str)
                except json.JSONDecodeError:
                    # Try to extract query directly
                    tool_input = {"query": input_str}

            return AgentAction.CALL_TOOL, thought, tool_name, tool_input

        # Default to thinking if no clear action
        return AgentAction.THINKING, thought, None, None

    def _execute_tool(self, tool_name: str, arguments: Dict) -> Neo4jResult:
        """Execute a tool via Neo4j client."""
        if not self.neo4j_client:
            return Neo4jResult(
                success=False,
                data=None,
                error="Neo4j client not provided"
            )

        if not self.neo4j_client.is_connected:
            return Neo4jResult(
                success=False,
                data=None,
                error="Neo4j client not connected. Call connect() first."
            )

        self._log(f"Executing tool: {tool_name}")
        self._log(f"Arguments: {json.dumps(arguments, indent=2)}")

        if tool_name == "get_schema":
            return self.neo4j_client.get_schema()

        elif tool_name == "run_cypher":
            query = arguments.get("query", "")
            return self.neo4j_client.run_cypher(query)

        elif tool_name == "write_cypher":
            query = arguments.get("query", "")
            return self.neo4j_client.write_cypher(query)

        else:
            return Neo4jResult(
                success=False,
                data=None,
                error=f"Unknown tool: {tool_name}"
            )

    def run(self, task: str, context: Optional[Dict] = None) -> AgentResult:
        """
        Run the agent on a task.

        Args:
            task: The task description
            context: Optional context (e.g., data model, sample data)

        Returns:
            AgentResult with success status, answer, and step history
        """
        self.history = []
        system_prompt = self._get_system_prompt(context)

        # Build initial prompt
        full_prompt = f"{system_prompt}\n\nUser task: {task}\n\nBegin:\n"

        self._log(f"Starting agent with task: {task}")
        self._log(f"Max iterations: {self.max_iterations}")

        for iteration in range(self.max_iterations):
            self._log(f"\n{'='*50}")
            self._log(f"Iteration {iteration + 1}/{self.max_iterations}")
            self._log(f"{'='*50}")

            try:
                # Get LLM response
                response = self._call_ollama_generate(full_prompt)
                self._log(f"LLM Response:\n{response[:500]}...")

                # Parse response
                action, thought, tool_name, tool_input = self._parse_agent_response(response)

                timestamp = datetime.now().isoformat()

                if action == AgentAction.FINAL_ANSWER:
                    final_answer = tool_input.get("answer", response) if tool_input else response

                    step = AgentStep(
                        timestamp=timestamp,
                        thought=thought,
                        action=action,
                        observation=final_answer
                    )
                    self.history.append(step)

                    if self.on_step:
                        self.on_step(step)

                    self._log(f"Final answer reached!")
                    return AgentResult(
                        success=True,
                        answer=final_answer,
                        steps=self.history,
                        total_iterations=iteration + 1
                    )

                elif action == AgentAction.CALL_TOOL:
                    # Execute tool
                    tool_result = self._execute_tool(tool_name, tool_input or {})

                    observation = str(tool_result.data) if tool_result.success else f"Error: {tool_result.error}"

                    step = AgentStep(
                        timestamp=timestamp,
                        thought=thought,
                        action=action,
                        tool_name=tool_name,
                        tool_input=tool_input,
                        observation=str(observation)[:1000]  # Truncate long observations
                    )
                    self.history.append(step)

                    if self.on_step:
                        self.on_step(step)

                    # Add observation to prompt for next iteration
                    full_prompt += f"\n{response}\nOBSERVATION: {observation}\n\nContinue:\n"

                else:
                    # Just thinking, continue
                    step = AgentStep(
                        timestamp=timestamp,
                        thought=thought,
                        action=action
                    )
                    self.history.append(step)

                    if self.on_step:
                        self.on_step(step)

                    full_prompt += f"\n{response}\n\nContinue (remember to use ACTION or FINAL_ANSWER):\n"

            except requests.exceptions.ConnectionError:
                error_msg = "Cannot connect to Ollama. Make sure it's running (ollama serve)"
                self._log(f"ERROR: {error_msg}")
                return AgentResult(
                    success=False,
                    answer="",
                    steps=self.history,
                    error=error_msg,
                    total_iterations=iteration + 1
                )

            except Exception as e:
                error_msg = str(e)
                self._log(f"ERROR: {error_msg}")

                step = AgentStep(
                    timestamp=datetime.now().isoformat(),
                    thought=f"Error occurred: {error_msg}",
                    action=AgentAction.ERROR
                )
                self.history.append(step)

                if self.on_step:
                    self.on_step(step)

                # Try to continue
                full_prompt += f"\nError occurred: {error_msg}\nPlease try a different approach:\n"

        # Max iterations reached
        self._log("Max iterations reached without final answer")
        return AgentResult(
            success=False,
            answer="Max iterations reached without completing the task.",
            steps=self.history,
            error="Max iterations reached",
            total_iterations=self.max_iterations
        )


def simple_chat(message: str, neo4j_client: Optional[Neo4jClient] = None,
                model: str = "hf.co/HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF:Q4_K_M",
                verbose: bool = True) -> str:
    """
    Simple chat function for quick interactions.

    Args:
        message: User message
        neo4j_client: Optional Neo4j client for database context
        model: Ollama model name
        verbose: Print prompts and responses to console

    Returns:
        Agent response string
    """
    # Get schema context if client is connected
    context = None
    if neo4j_client and neo4j_client.is_connected:
        schema_result = neo4j_client.get_schema()
        if schema_result.success:
            context = {"database_schema": schema_result.data}

    # Build simple prompt
    system = """You are a helpful assistant for Neo4j data modeling.
Answer questions about knowledge graphs, data modeling, and Neo4j.
If asked to write Cypher queries, provide them in code blocks."""

    if context:
        system += f"\n\nDatabase context: {json.dumps(context, indent=2)}"

    prompt = f"{system}\n\nUser: {message}\n\nAssistant:"

    if verbose:
        print(f"\n{'='*80}")
        print(f"[simple_chat] === PROMPT START ===")
        print(prompt)
        print(f"[simple_chat] === PROMPT END === ({len(prompt)} chars)")
        print(f"{'='*80}")

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 500}
            },
            timeout=60
        )
        response.raise_for_status()
        result = response.json().get("response", "")

        if verbose:
            print(f"[simple_chat] === RESPONSE START ===")
            print(result)
            print(f"[simple_chat] === RESPONSE END === ({len(result)} chars)")
            print(f"{'='*80}")

        return result

    except requests.exceptions.ConnectionError:
        return "Error: Cannot connect to Ollama. Make sure it's running (ollama serve)"
    except Exception as e:
        return f"Error: {str(e)}"


# Convenience function for quick testing
def test_agent():
    """Test the agent with a simple task."""
    # Create and connect Neo4j client
    client = Neo4jClient()

    try:
        client.connect()

        # Create agent
        agent = SmolLM2Agent(neo4j_client=client, verbose=True)

        # Run a simple task
        result = agent.run(
            "Get the current schema of the Neo4j database and tell me what nodes exist."
        )

        print("\n" + "="*60)
        print("AGENT RESULT")
        print("="*60)
        print(f"Success: {result.success}")
        print(f"Answer: {result.answer}")
        print(f"Iterations: {result.total_iterations}")
        print(f"Steps: {len(result.steps)}")

        for i, step in enumerate(result.steps):
            print(f"\nStep {i+1}:")
            print(f"  Thought: {step.thought[:100]}...")
            print(f"  Action: {step.action.value}")
            if step.tool_name:
                print(f"  Tool: {step.tool_name}")

    finally:
        client.close()


if __name__ == "__main__":
    test_agent()
