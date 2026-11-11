//! Camera management HTTP API. Each camera is a child operating-system process.
use axum::{extract::{Path, State}, http::StatusCode, routing::get, Json, Router};
use serde_json::{json, Value};
use std::sync::Arc;
use crate::{process_manager::{CameraProcessManager, CameraProcessStatus, StopResult}, config::CameraConfig};

type ErrorResponse = (StatusCode, Json<Value>);
fn bad_request(error: impl ToString) -> ErrorResponse {
    (StatusCode::BAD_REQUEST, Json(json!({"error": error.to_string()})))
}
fn internal_error(error: impl ToString) -> ErrorResponse {
    (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": error.to_string()})))
}

pub fn router(manager: Arc<CameraProcessManager>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/api/cameras", get(cameras).post(add_camera))
        .route("/api/cameras/{id}", axum::routing::delete(delete_camera))
        .route("/api/cameras/{id}/restart", axum::routing::post(restart_camera))
        .with_state(manager)
}

async fn health() -> Json<Value> {
    Json(json!({"status": "ok", "service": "rustlpr-manager"}))
}
async fn cameras(State(manager): State<Arc<CameraProcessManager>>) -> Result<Json<Vec<CameraProcessStatus>>, ErrorResponse> {
    manager.list().await.map(Json).map_err(internal_error)
}
async fn add_camera(
    State(manager): State<Arc<CameraProcessManager>>,
    Json(camera): Json<CameraConfig>,
) -> Result<(StatusCode, Json<CameraProcessStatus>), ErrorResponse> {
    let status = manager.start(camera).await.map_err(bad_request)?;
    Ok((StatusCode::CREATED, Json(status)))
}
async fn delete_camera(
    State(manager): State<Arc<CameraProcessManager>>,
    Path(id): Path<String>,
) -> Result<Json<StopResult>, ErrorResponse> {
    manager.stop(&id).await.map_err(internal_error)?
        .map(Json)
        .ok_or_else(|| (StatusCode::NOT_FOUND, Json(json!({"error": "camera not found"}))))
}
async fn restart_camera(
    State(manager): State<Arc<CameraProcessManager>>,
    Path(id): Path<String>,
) -> Result<Json<CameraProcessStatus>, ErrorResponse> {
    manager.restart(&id).await.map_err(internal_error)?
        .map(Json)
        .ok_or_else(|| (StatusCode::NOT_FOUND, Json(json!({"error": "camera not found"}))))
}
