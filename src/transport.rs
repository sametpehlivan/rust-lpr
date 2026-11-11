//! Outbound TCP protocol: one JSON TrackingEvent per newline (NDJSON).
//! This is best-effort delivery: if TCP breaks, the in-flight event is retried;
//! no remote acknowledgment or disk spool exists, so process crashes can lose events.
use crate::domain::TrackingEvent;
use tokio::{io::AsyncWriteExt, net::TcpStream, sync::mpsc, time::{sleep,timeout,Duration}};
use tracing::{info,warn,error};

pub async fn run_tcp_sender(destination:String,mut receiver:mpsc::Receiver<TrackingEvent>) {
    let mut pending:Option<TrackingEvent>=None;
    loop {
        if receiver.is_closed() && receiver.is_empty() && pending.is_none() {return;}
        let mut stream=match timeout(Duration::from_secs(5),TcpStream::connect(&destination)).await {
            Ok(Ok(s)) => {info!(target=%destination,"TCP event transport connected");s}
            other => {
                warn!(target=%destination,result=?other,"TCP unavailable; retrying");
                sleep(Duration::from_secs(3)).await;
                continue;
            }
        };
        let _=stream.set_nodelay(true);
        loop {
            let event=if let Some(event)=pending.take(){event}
                else if let Some(event)=receiver.recv().await{event}
                else {info!("TCP event transport closing");return};
            let mut line=match serde_json::to_vec(&event) {
                Ok(data)=>data,
                Err(e)=>{error!(error=%e,"event serialization failed");continue;}
            };
            line.push(b'\n');
            let result=timeout(Duration::from_secs(5),stream.write_all(&line)).await;
            if !matches!(result,Ok(Ok(()))) {
                warn!(event_id=%event.event_id,"TCP write failure; retaining event until reconnect");
                pending=Some(event);
                break;
            }
        }
        sleep(Duration::from_secs(2)).await;
    }
}
