"""
PF Recovery Agent – CLI entry point.

Usage examples
--------------
# Interactive mode (prompts for outage details):
    python main.py run

# Provide outage via command-line flags:
    python main.py run \\
        --outage-id ICM-12345 \\
        --title "PFGateway 502 errors in East US" \\
        --services PFGateway PFOrchestrator \\
        --regions eastus \\
        --symptoms "HTTP 502" "health probe failing" \\
        --severity SEV1

# Provide outage via JSON file:
    python main.py run --from-file outage.json

# Demo mode (no OpenAI key required):
    python main.py demo
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import typer
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    name="pf-agent",
    help="PF Recovery Agent – faster RTO during Azure PF outages.",
    add_completion=False,
)
console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _print_response(response) -> None:
    """Pretty-print an AgentResponse to the terminal."""
    rprint(Panel(f"[bold cyan]Outage:[/bold cyan] {response.outage_id}", expand=False))
    rprint(Panel(f"[bold yellow]Summary[/bold yellow]\n\n{response.summary}"))

    if response.estimated_rto_minutes:
        rprint(
            f"\n[bold]Estimated RTO:[/bold] ~{response.estimated_rto_minutes} minutes "
            f"(based on similar historical incidents)"
        )

    if response.dependency_risks:
        rprint("\n[bold red]Dependency Risks (Blast Radius)[/bold red]")
        for risk in response.dependency_risks:
            rprint(f"  • {risk}")

    if response.recommended_actions:
        rprint("\n[bold green]Recommended Actions[/bold green]")
        for i, action in enumerate(response.recommended_actions, 1):
            rprint(f"  {i}. {action}")

    if response.relevant_rcas:
        rprint("\n[bold]Relevant Historical RCAs[/bold]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Title")
        table.add_column("Date", no_wrap=True)
        table.add_column("RTO (min)", justify="right")
        for rca in response.relevant_rcas[:3]:
            table.add_row(
                rca.get("id", ""),
                rca.get("title", ""),
                rca.get("date", ""),
                str(rca.get("time_to_mitigate_minutes", "")),
            )
        console.print(table)

    if response.relevant_tsgs:
        rprint("\n[bold]Relevant TSGs[/bold]")
        for tsg in response.relevant_tsgs[:3]:
            rprint(f"\n[underline]{tsg.get('id')} – {tsg.get('title')}[/underline]")
            rprint("[dim]Diagnostic Steps:[/dim]")
            for step in tsg.get("diagnostic_steps", [])[:3]:
                rprint(f"  → {step}")
            rprint("[dim]Mitigation Steps:[/dim]")
            for step in tsg.get("mitigation_steps", [])[:3]:
                rprint(f"  ✓ {step}")
            if tsg.get("escalation_path"):
                rprint(f"[dim]Escalation:[/dim] {tsg['escalation_path']}")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def run(
    outage_id: str = typer.Option(None, "--outage-id", help="ICM / incident identifier"),
    title: str = typer.Option(None, "--title", help="Short outage title"),
    services: list[str] = typer.Option(None, "--services", help="Affected PF service names"),
    regions: list[str] = typer.Option(None, "--regions", help="Affected Azure regions"),
    clusters: list[str] = typer.Option(None, "--clusters", help="Affected cluster names"),
    severity: str = typer.Option("SEV1", "--severity", help="SEV0/SEV1/SEV2/SEV3"),
    symptoms: list[str] = typer.Option(None, "--symptoms", help="Observed symptoms"),
    context: str = typer.Option("", "--context", help="Additional free-text context"),
    from_file: Path = typer.Option(None, "--from-file", help="Load outage from JSON file"),
    output_json: bool = typer.Option(False, "--json", help="Output raw JSON response"),
) -> None:
    """Analyse an active PF outage and generate a recovery plan."""
    from pf_recovery_agent.agent import run_agent
    from pf_recovery_agent.models import OutageEvent, OutageSeverity

    if from_file:
        with open(from_file, encoding="utf-8") as fh:
            data = json.load(fh)
        outage = OutageEvent(**data)
    else:
        # Fill in any missing values interactively
        if not outage_id:
            outage_id = typer.prompt("Outage ID (e.g. ICM-12345)", default="ICM-UNKNOWN")
        if not title:
            title = typer.prompt("Outage title")
        if not services:
            svc_input = typer.prompt("Affected services (comma-separated)")
            services = [s.strip() for s in svc_input.split(",")]

        outage = OutageEvent(
            id=outage_id,
            title=title,
            affected_services=services or [],
            affected_regions=regions or [],
            affected_clusters=clusters or [],
            severity=OutageSeverity(severity),
            symptoms=symptoms or [],
            additional_context=context,
            start_time=datetime.utcnow(),
        )

    console.print(f"\n[bold cyan]PF Recovery Agent[/bold cyan] analysing outage [bold]{outage.id}[/bold]…\n")

    try:
        response = run_agent(outage)
    except RuntimeError as exc:
        rprint(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1)

    if output_json:
        print(response.model_dump_json(indent=2))
    else:
        _print_response(response)


@app.command()
def demo() -> None:
    """
    Run a demonstration using sample data without requiring an OpenAI API key.
    Exercises the knowledge-base tools directly and prints their output.
    """
    from pf_recovery_agent.tools.dependency_tool import get_service_dependencies
    from pf_recovery_agent.tools.rca_tool import search_rcas
    from pf_recovery_agent.tools.tsg_tool import search_tsgs

    rprint(Panel("[bold cyan]PF Recovery Agent – Demo Mode[/bold cyan]", expand=False))
    rprint("\nSimulating outage: PFGateway + PFOrchestrator failures in East US\n")

    services = ["PFGateway", "PFOrchestrator"]
    symptoms = ["HTTP 502", "health probe failing", "TLS error"]
    keywords = ["tls", "gateway", "certificate"]

    rprint("[bold yellow]1. Service Dependencies[/bold yellow]")
    deps = get_service_dependencies(services)
    rprint(f"  Blast radius: {', '.join(deps['blast_radius'])}")
    rprint(f"  Tier-1 risks: {', '.join(deps['tier1_risks']) or 'none'}")

    rprint("\n[bold yellow]2. Matching RCAs[/bold yellow]")
    rcas = search_rcas(services, keywords=keywords, max_results=3)
    for rca in rcas:
        rprint(f"  [{rca['id']}] {rca['title']} (RTO: {rca['time_to_mitigate_minutes']}min, score: {rca['relevance_score']})")

    rprint("\n[bold yellow]3. Relevant TSGs[/bold yellow]")
    tsgs = search_tsgs(services, symptoms=symptoms, max_results=3)
    for tsg in tsgs:
        rprint(f"  [{tsg['id']}] {tsg['title']} (score: {tsg['relevance_score']})")
        rprint(f"    Escalation: {tsg['escalation_path']}")

    rprint("\n[bold green]Demo complete. Set OPENAI_API_KEY or AZURE_OPENAI_* env vars and run `python main.py run` for full AI analysis.[/bold green]")


@app.command()
def list_services() -> None:
    """List all known PF services in the dependency graph."""
    from pf_recovery_agent.knowledge_base.loader import load_service_dependencies

    deps = load_service_dependencies()
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Service", style="cyan")
    table.add_column("Cluster")
    table.add_column("Tier", justify="center")
    table.add_column("Description")

    seen: set[str] = set()
    for dep in sorted(deps, key=lambda d: (d.tier, d.service_name)):
        key = f"{dep.service_name}:{dep.cluster}"
        if key not in seen:
            seen.add(key)
            table.add_row(dep.service_name, dep.cluster, str(dep.tier), dep.description[:60])
    console.print(table)


@app.command("list-rcas")
def list_rcas() -> None:
    """List all RCA records in the knowledge base."""
    from pf_recovery_agent.knowledge_base.loader import load_rcas

    rcas = load_rcas()
    table = Table(show_header=True, header_style="bold yellow")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Title")
    table.add_column("Date", no_wrap=True)
    table.add_column("Services")
    table.add_column("RTO (min)", justify="right")
    for rca in sorted(rcas, key=lambda r: r.date, reverse=True):
        table.add_row(
            rca.id,
            rca.title[:50],
            rca.date,
            ", ".join(rca.affected_services[:2]),
            str(rca.time_to_mitigate_minutes),
        )
    console.print(table)


@app.command("list-tsgs")
def list_tsgs() -> None:
    """List all TSG entries in the knowledge base."""
    from pf_recovery_agent.knowledge_base.loader import load_tsgs

    tsgs = load_tsgs()
    table = Table(show_header=True, header_style="bold green")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Title")
    table.add_column("Applicable Services")
    table.add_column("Escalation")
    for tsg in tsgs:
        table.add_row(
            tsg.id,
            tsg.title[:50],
            ", ".join(tsg.applicable_services[:2]),
            tsg.escalation_path[:40],
        )
    console.print(table)


if __name__ == "__main__":
    app()
