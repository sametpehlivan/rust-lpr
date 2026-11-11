use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}
impl BBox {
    pub fn width(self) -> f32 { (self.x2 - self.x1).max(0.0) }
    pub fn height(self) -> f32 { (self.y2 - self.y1).max(0.0) }
    pub fn center(self) -> (f32, f32) { ((self.x1+self.x2)/2.0, (self.y1+self.y2)/2.0) }
    pub fn iou(self, other: Self) -> f32 {
        let w = (self.x2.min(other.x2)-self.x1.max(other.x1)).max(0.0);
        let h = (self.y2.min(other.y2)-self.y1.max(other.y1)).max(0.0);
        let intersection = w*h;
        let union = self.width()*self.height() + other.width()*other.height()-intersection;
        if union <= 0.0 {0.0} else {intersection/union}
    }
    pub fn distance(self, other: Self) -> f32 {
        let (ax,ay)=self.center(); let (bx,by)=other.center();
        ((ax-bx).powi(2)+(ay-by).powi(2)).sqrt()
    }
    pub fn clamp(self, w: u32, h: u32) -> Self {
        Self { x1: self.x1.clamp(0.0,w as f32), y1: self.y1.clamp(0.0,h as f32),
            x2: self.x2.clamp(0.0,w as f32), y2: self.y2.clamp(0.0,h as f32) }
    }
}

#[derive(Debug, Clone)]
pub struct PlateObservation {
    pub bbox: BBox,
    pub detection_score: f32,
    pub plate: String,
    pub ocr_score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlateCandidate {
    pub value: String,
    pub score: f32,
    pub occurrences: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackingValues {
    #[serde(rename = "Plates")]
    pub plates: Vec<PlateCandidate>,
    #[serde(rename = "Direction")]
    pub direction: [f32; 2],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackingEvent {
    pub event_id: Uuid,
    pub event_type: String,
    pub source_id: String,
    pub tracked_id: Uuid,
    pub timestamp_utc: DateTime<Utc>,
    pub bbox_xyxy: [f32;4],
    pub plate: String,
    pub confidence: f32,
    pub observations: usize,
    pub tracking_values: TrackingValues,
}
