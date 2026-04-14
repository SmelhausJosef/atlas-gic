from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from atlas_admin.job_registry import JobRegistry, build_default_registry
from atlas_admin.services import DashboardService


def create_app(
    *,
    service: DashboardService | None = None,
    job_registry: JobRegistry | None = None,
) -> FastAPI:
    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "configs" / "base.yaml"
    dashboard_service = service or DashboardService(project_root=project_root, config_path=config_path)
    registry = job_registry or build_default_registry(project_root=project_root, config_path=config_path)
    templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))

    app = FastAPI(title="ATLAS Admin")
    app.mount(
        "/static",
        StaticFiles(directory=str(Path(__file__).resolve().parent / "static")),
        name="static",
    )
    app.state.project_root = project_root
    app.state.service = dashboard_service
    app.state.registry = registry
    app.state.templates = templates

    def _normalize_int(value: str | None) -> int | None:
        if value is None or not value.strip():
            return None
        return int(value)

    @app.get("/", response_class=HTMLResponse)
    async def dashboard_home(request: Request) -> HTMLResponse:
        service_ref: DashboardService = request.app.state.service
        registry_ref: JobRegistry = request.app.state.registry
        runs = service_ref.list_runs(registry_ref)[:10]
        active_runs = [item for item in runs if item.status == "running"]
        latest_baseline = next((item for item in runs if item.run_type == "backtest"), None)
        latest_research = next((item for item in runs if item.run_type == "research"), None)
        all_runs = service_ref.list_runs(registry_ref)
        run_counts = {
            "total": len(all_runs),
            "backtests": sum(1 for item in all_runs if item.run_type == "backtest"),
            "research": sum(1 for item in all_runs if item.run_type == "research"),
            "running": sum(1 for item in all_runs if item.status == "running"),
            "failed": sum(1 for item in all_runs if item.status == "failed"),
        }
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "dataset": service_ref.get_dataset_overview(),
                "strategies": service_ref.get_available_strategies(),
                "runs": runs,
                "active_runs": active_runs,
                "latest_baseline": latest_baseline,
                "latest_research": latest_research,
                "run_counts": run_counts,
            },
        )

    @app.get("/runs", response_class=HTMLResponse)
    async def runs_list(request: Request, kind: str | None = None) -> HTMLResponse:
        service_ref: DashboardService = request.app.state.service
        registry_ref: JobRegistry = request.app.state.registry
        return templates.TemplateResponse(
            request=request,
            name="runs.html",
            context={"runs": service_ref.list_runs(registry_ref, kind=kind), "kind": kind},
        )

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    async def run_detail(request: Request, run_id: str) -> HTMLResponse:
        service_ref: DashboardService = request.app.state.service
        registry_ref: JobRegistry = request.app.state.registry
        detail = service_ref.get_run_detail(run_id, registry_ref)
        if detail is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return templates.TemplateResponse(request=request, name="run_detail.html", context={"detail": detail})

    @app.get("/api/heartbeat/{run_id}", response_class=JSONResponse)
    async def heartbeat_json(request: Request, run_id: str) -> JSONResponse:
        service_ref: DashboardService = request.app.state.service
        registry_ref: JobRegistry = request.app.state.registry
        payload = service_ref.get_heartbeat_payload(run_id, registry_ref)
        if payload is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return JSONResponse(payload)

    @app.get("/partials/active-sessions", response_class=HTMLResponse)
    async def active_sessions_partial(request: Request) -> HTMLResponse:
        service_ref: DashboardService = request.app.state.service
        registry_ref: JobRegistry = request.app.state.registry
        active_runs = [item for item in service_ref.list_runs(registry_ref) if item.status == "running"][:6]
        return templates.TemplateResponse(
            request=request,
            name="partials/active_sessions.html",
            context={"active_runs": active_runs},
        )

    @app.get("/partials/heartbeat/{run_id}", response_class=HTMLResponse)
    async def heartbeat_partial(request: Request, run_id: str) -> HTMLResponse:
        service_ref: DashboardService = request.app.state.service
        registry_ref: JobRegistry = request.app.state.registry
        detail = service_ref.get_run_detail(run_id, registry_ref)
        if detail is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return templates.TemplateResponse(
            request=request,
            name="partials/heartbeat.html",
            context={"detail": detail},
        )

    @app.get("/partials/log-preview/{run_id}", response_class=HTMLResponse)
    async def log_preview_partial(request: Request, run_id: str) -> HTMLResponse:
        service_ref: DashboardService = request.app.state.service
        registry_ref: JobRegistry = request.app.state.registry
        detail = service_ref.get_run_detail(run_id, registry_ref)
        if detail is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return templates.TemplateResponse(
            request=request,
            name="partials/log_preview.html",
            context={"detail": detail},
        )

    @app.post("/actions/backtest")
    async def start_backtest(
        request: Request,
        config_path_value: str = Form(default="configs/base.yaml"),
        strategy: str = Form(default="btc_mean_reversion_v1"),
    ) -> RedirectResponse:
        registry_ref: JobRegistry = request.app.state.registry
        config_path_resolved = (request.app.state.project_root / config_path_value).resolve()
        registry_ref.launch_backtest(config_path=config_path_resolved, strategy=strategy)
        return RedirectResponse(url="/", status_code=303)

    @app.post("/actions/research")
    async def start_research(
        request: Request,
        config_path_value: str = Form(default="configs/base.yaml"),
        baseline: str = Form(default="btc_mean_reversion_v1"),
        candidate_budget: str | None = Form(default=None),
        max_generations: str | None = Form(default=None),
    ) -> RedirectResponse:
        registry_ref: JobRegistry = request.app.state.registry
        config_path_resolved = (request.app.state.project_root / config_path_value).resolve()
        registry_ref.launch_research(
            config_path=config_path_resolved,
            baseline=baseline,
            candidate_budget=_normalize_int(candidate_budget),
            max_generations=_normalize_int(max_generations),
            use_llm=True,
        )
        return RedirectResponse(url="/", status_code=303)

    @app.post("/actions/stop/{run_id}")
    async def stop_run(request: Request, run_id: str) -> RedirectResponse:
        registry_ref: JobRegistry = request.app.state.registry
        registry_ref.stop(run_id)
        return RedirectResponse(url=f"/runs/{run_id}", status_code=303)

    @app.get("/artifacts/{run_id}/{filename}")
    async def artifact_download(request: Request, run_id: str, filename: str) -> FileResponse:
        service_ref: DashboardService = request.app.state.service
        run_detail_value = service_ref.get_run_detail(run_id, request.app.state.registry)
        if run_detail_value is None or filename not in run_detail_value.artifact_files:
            raise HTTPException(status_code=404, detail="Artifact not found")
        artifact_path = service_ref.artifacts_dir / run_id / filename
        return FileResponse(artifact_path)

    return app
