import sys
from typing import Optional, Union, List
from rich.console import Console, Group
from rich.panel import Panel
from rich.live import Live
from rich.prompt import Prompt
from rich.text import Text
from groq import Groq # Ensure you have the latest version of groq installed, at least 0.23.1, as the executed_tools field is new.

class CompoundCLI:
    def __init__(self):
        self.console = Console()
        self.client = Groq() # API key by default comes from GROQ_API_KEY in environment variables. You can also pass in explicitly with Groq(api_key=os.environ["GROQ_API_KEY"])
        self.max_stream_height = max(10, self.console.height // 4)

    def get_user_input(self) -> Optional[str]:
        try:
            return Prompt.ask(">> ")
        except KeyboardInterrupt:
            self.console.print("\n[red]Exiting...[/red]")
            sys.exit(0)

    def format_tools(self, tools: Union[List[str], str]) -> str:
        if isinstance(tools, list):
            formatted_tools = []
            for tool in tools:
                tool_str = f"- {tool.type}"
                if tool.output:
                    tool_str += f"\n  Output: {tool.output}"
                formatted_tools.append(tool_str)
            return "\n".join(formatted_tools)
        return str(tools)

    def create_stream_panel(self, reasoning: str = "", content: str = "", executed_tools: Union[List[str], str] = "", tool_in_progress: bool = False) -> tuple[Optional[Panel], Optional[Panel], Optional[Panel]]:
        working_panel = tools_panel = answer_panel = None
        
        if reasoning:
            lines = reasoning.split('\n')
            panel_height = min(len(lines) + 2, self.max_stream_height)
            working_panel = Panel(
                Text('\n'.join(lines[-self.max_stream_height:]), style="yellow"),
                title="[bold]Working...[/bold]",
                border_style="blue",
                padding=(1, 2),
                height=panel_height
            )

        if tool_in_progress:
            tools_panel = Panel(
                Text("Executing tool...", style="blue"),
                border_style="blue",
                padding=(1, 2)
            )
            
        elif executed_tools:
            tools_text = Text()
            for tool in executed_tools:
                tools_text.append(f"{tool.type}\n", style="bold blue")
                if tool.output:
                    tools_text.append(f"{tool.output}\n", style="blue")
            
            panel_height = min(len(tools_text.split('\n')) + 2, self.max_stream_height)
            tools_panel = Panel(
                tools_text,
                title="[bold]Tools Used[/bold]",
                border_style="blue",
                padding=(1, 2),
                height=panel_height
            )

        if content:
            answer_panel = Text(content+"\n", style="bold green")
        
        return working_panel, tools_panel, answer_panel

    def stream_response(self, messages):
        with Live(auto_refresh=True) as live:
            full_response = {"reasoning": "", "content": "", "executed_tools": []}
            tool_in_progress = False
            
            # Stream the response from compound-beta
            for chunk in self.client.chat.completions.create(
                messages=messages,
                model="compound-beta",
                stream=True
            ):
                if chunk.choices[0].delta.reasoning:
                    full_response["reasoning"] += chunk.choices[0].delta.reasoning
                if chunk.choices[0].delta.content:
                    full_response["content"] += chunk.choices[0].delta.content
                if chunk.choices[0].delta.executed_tools:
                    tools = chunk.choices[0].delta.executed_tools
                    if isinstance(tools, list):
                        # When streaming, when a tool begins being executed, we get the tool object with an empty output.
                        # When the tool is finished executing, we get the tool object again with the resulting output.
                        if tools[0].output:
                            full_response["executed_tools"] = tools
                            tool_in_progress = False
                        else:
                            tool_in_progress = True

                working_panel, tools_panel, answer_panel = self.create_stream_panel(
                    full_response["reasoning"],
                    full_response["content"],
                    full_response["executed_tools"],
                    tool_in_progress
                )
                
                panels = [p for p in (working_panel, tools_panel, answer_panel) if p]
                live.update(Group(*panels) if panels else "")

        # Add the assistant's response to messages after streaming is complete
        messages.append({"role": "assistant", "content": full_response["content"]})
    
    def run(self):
        messages = []
        while True:
            user_input = self.get_user_input()
            if not user_input:
                continue
            messages.append({"role": "user", "content": user_input})
            self.stream_response(messages)

if __name__ == "__main__":
    cli = CompoundCLI()
    cli.run() 