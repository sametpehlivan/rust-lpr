//! Per-camera plate tracking. Follows NetLpr's PlateTrackingService policy:
//! IoU >= .15, then center distance <= 120px, then weighted OCR similarity >= 80.
//! This tracks the detected plate rectangle; the current C# project uses EmptyKalmanFilter.
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use uuid::Uuid;
use crate::{config::CameraConfig, domain::{BBox, PlateCandidate, PlateObservation, TrackingEvent, TrackingValues}};

struct Track {
    id: Uuid,
    last: BBox,
    first: BBox,
    observations: Vec<PlateObservation>,
    last_seen: DateTime<Utc>,
    misses: u32,
}

pub struct PlateTracker {
    camera: CameraConfig,
    tracks: Vec<Track>,
}

impl PlateTracker {
    pub fn new(camera: CameraConfig) -> Self { Self { camera, tracks: Vec::new() } }

    pub fn update(&mut self, detections: Vec<PlateObservation>, timestamp: DateTime<Utc>) -> Vec<TrackingEvent> {
        let valid: Vec<PlateObservation> = detections.into_iter()
            .filter(|d| d.ocr_score >= 0.8 && d.plate.len() >= 5)
            .collect();
        for track in &mut self.tracks { track.misses += 1; }

        let mut occupied_tracks = vec![false; self.tracks.len()];
        let mut occupied_dets = vec![false; valid.len()];
        // Three consecutive greedy assignment passes, matching the C# implementation.
        for stage in 0..3 {
            loop {
                let mut best: Option<(usize,usize,f32)> = None;
                for (ti,t) in self.tracks.iter().enumerate() {
                    if occupied_tracks[ti] { continue; }
                    for (di,d) in valid.iter().enumerate() {
                        if occupied_dets[di] { continue; }
                        let score = match stage {
                            0 => { let iou = t.last.iou(d.bbox); if iou >= 0.15 {iou} else {-1.0} },
                            1 => { let distance=t.last.distance(d.bbox); if distance <= 120.0 { 120.0 - distance } else {-1.0} },
                            _ => { let previous=t.observations.last().map(|d| d.plate.as_str()).unwrap_or("");
                                let similarity=weighted_similarity(previous,&d.plate);
                                if similarity >= 80.0 {similarity} else {-1.0} }
                        };
                        if score > 0.0 && best.map_or(true, |b| score > b.2) { best=Some((ti,di,score)); }
                    }
                }
                if let Some((ti,di,_))=best {
                    occupied_tracks[ti]=true;
                    occupied_dets[di]=true;
                    let track=&mut self.tracks[ti];
                    track.last=valid[di].bbox;
                    track.observations.push(valid[di].clone());
                    track.misses=0;
                    track.last_seen=timestamp;
                } else {break;}
            }
        }
        for (i,det) in valid.into_iter().enumerate() {
            if occupied_dets[i] || !within_polygon(det.bbox,self.camera.width,self.camera.height,&self.camera.polygon) {continue;}
            self.tracks.push(Track {
                id:Uuid::new_v4(), last:det.bbox, first:det.bbox,
                observations:vec![det],last_seen:timestamp,misses:0,
            });
        }
        self.collect_finished(timestamp,false)
    }

    pub fn expire_idle(&mut self, now: DateTime<Utc>) -> Vec<TrackingEvent> { self.collect_finished(now,false) }
    pub fn flush(&mut self, now: DateTime<Utc>) -> Vec<TrackingEvent> { self.collect_finished(now,true) }
    fn collect_finished(&mut self, now: DateTime<Utc>, all: bool) -> Vec<TrackingEvent> {
        let mut completed=Vec::new();
        let mut pos=0;
        while pos < self.tracks.len() {
            let track=&self.tracks[pos];
            let expired= all || track.misses > self.camera.max_track_age_frames
                || (now-track.last_seen).num_seconds() >= self.camera.max_track_idle_seconds as i64;
            if expired {
                let track=self.tracks.swap_remove(pos);
                completed.push(to_event(&self.camera.id,track,now));
            } else {pos+=1;}
        }
        completed
    }
}

fn to_event(camera_id:&str,track:Track,now:DateTime<Utc>)->TrackingEvent {
    let mut frequencies:HashMap<String,(usize,f32)> = HashMap::new();
    for det in &track.observations {
        let record=frequencies.entry(det.plate.clone()).or_insert((0,0.0));
        record.0+=1;
        record.1+=det.ocr_score;
    }
    let longest=frequencies.keys().map(String::len).max().unwrap_or(1).max(1);
    let count=track.observations.len().max(1);
    let mut plates:Vec<PlateCandidate>=frequencies.into_iter().map(|(plate,(n,sum))| {
        let avg=sum/n as f32;
        let score=0.2*(plate.len() as f32 / longest as f32) + 0.6*(n as f32 / count as f32) + 0.2*avg;
        PlateCandidate {value:plate,score,occurrences:n}
    }).collect();
    plates.sort_by(|a,b| b.score.total_cmp(&a.score));
    let first=plates.first();
    let direction=[track.last.x1-track.first.x1,track.last.y1-track.first.y1];
    TrackingEvent {
        event_id:Uuid::new_v4(),event_type:"TRACK_COMPLETED".into(),source_id:camera_id.to_string(),
        tracked_id:track.id,timestamp_utc:now,
        bbox_xyxy:[track.last.x1,track.last.y1,track.last.x2,track.last.y2],
        plate:first.map(|p|p.value.clone()).unwrap_or_default(),
        confidence:first.map_or(0.0,|p|p.score),
        observations:track.observations.len(),
        tracking_values:TrackingValues{plates,direction},
    }
}

fn within_polygon(bbox:BBox,w:u32,h:u32,points:&[[f32;2]])->bool {
    if points.is_empty(){return true;}
    [(bbox.x1,bbox.y1),(bbox.x2,bbox.y1),(bbox.x2,bbox.y2),(bbox.x1,bbox.y2)].iter()
        .any(|&(x,y)| point_in_polygon(x/w as f32,y/h as f32,points))
}
fn point_in_polygon(x:f32,y:f32,poly:&[[f32;2]])->bool {
    let mut inside=false;
    let mut j=poly.len()-1;
    for i in 0..poly.len() {
        let (xi,yi)=(poly[i][0],poly[i][1]); let (xj,yj)=(poly[j][0],poly[j][1]);
        if (yi>y)!=(yj>y) && x < (xj-xi)*(y-yi)/(yj-yi)+xi {inside = !inside;}
        j=i;
    }
    inside
}

fn char_penalty(a:char,b:char)->f32 {
    if a==b{return 0.0;}
    // Common ambiguities from NetLpr's weighted plate comparison.
    match if a<b {(a,b)}else{(b,a)} {
        ('0','O') | ('1','I') => 0.25,
        ('8','B')|('2','Z')|('5','S')|('O','Q')=>0.30,
        ('6','G')|('I','L')|('E','F')=>0.35,
        ('1','7')|('0','C')|('4','A')=>0.40,
        ('7','Z')|('9','Q')|('1','J')=>0.45,
        _=>1.0
    }
}
fn weighted_similarity(a:&str,b:&str)->f32 {
    let aa:Vec<char>=a.chars().collect(); let bb:Vec<char>=b.chars().collect();
    if aa.is_empty() && bb.is_empty(){return 100.0;}
    let mut prev:Vec<f32>=(0..=bb.len()).map(|i|i as f32).collect();
    for (i,&ca) in aa.iter().enumerate(){
        let mut next=vec![0.0;bb.len()+1];next[0]=(i+1) as f32;
        for (j,&cb) in bb.iter().enumerate(){
            next[j+1]=(prev[j+1]+1.0).min(next[j]+1.0).min(prev[j]+char_penalty(ca,cb));
        }
        prev=next;
    }
    (100.0*(1.0-prev[bb.len()]/aa.len().max(bb.len()) as f32)).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn config()->CameraConfig {CameraConfig {
        id:"cam1".into(),
        rtsp_url:"rtsp://localhost/test".into(),
        width:1280,
        height:720,
        fps:5,
        roi:None,
        polygon:vec![],max_track_age_frames:1,max_track_idle_seconds:60,
        onnx_sessions: Default::default(),
    }}
    fn observation(plate:&str,x:f32)->PlateObservation { PlateObservation {
        bbox:BBox{x1:x,y1:80.0,x2:x+80.0,y2:120.0},plate:plate.into(),ocr_score:0.95,detection_score:0.9,
    }}
    #[test] fn repeated_detections_are_one_track() {
        let mut tracker=PlateTracker::new(config());let now=Utc::now();
        assert!(tracker.update(vec![observation("06ABC123",100.0)],now).is_empty());
        assert!(tracker.update(vec![observation("06ABC123",110.0)],now).is_empty());
        assert!(tracker.update(vec![],now).is_empty());
        let out=tracker.update(vec![],now);
        assert_eq!(out.len(),1);assert_eq!(out[0].plate,"06ABC123");assert_eq!(out[0].observations,2);
    }
    #[test] fn fuzzy_plate_matching() {assert!(weighted_similarity("06ABC123","06ABC12B")>80.0);}
    #[test] fn iou_and_polygon(){
        assert!(BBox{x1:0.0,y1:0.0,x2:10.0,y2:10.0}.iou(BBox{x1:5.0,y1:5.0,x2:15.0,y2:15.0})>0.1);
        assert!(point_in_polygon(0.5,0.5,&[[0.1,0.1],[0.9,0.1],[0.9,0.9],[0.1,0.9]]));
        assert!(!point_in_polygon(0.0,0.0,&[[0.1,0.1],[0.9,0.1],[0.9,0.9],[0.1,0.9]]));
    }
}
