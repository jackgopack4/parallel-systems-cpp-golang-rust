//!
//! coordinator.rs
//! Implementation of 2PC coordinator
//!
extern crate log;
extern crate stderrlog;
extern crate rand;
extern crate ipc_channel;

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

use coordinator::ipc_channel::ipc::IpcSender as Sender;
use coordinator::ipc_channel::ipc::IpcReceiver as Receiver;
use coordinator::ipc_channel::ipc::TryRecvError;
use coordinator::ipc_channel::ipc::channel;

use message;
use message::MessageType;
use message::ProtocolMessage;
use message::RequestStatus;
use oplog;
use tpcoptions;

/// CoordinatorState
/// States for 2PC state machine
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoordinatorState {
    Quiescent,
    ReceivedRequest,
    ProposalSent,
    ReceivedVotesAbort,
    ReceivedVotesCommit,
    SentGlobalDecision
}
#[derive(Debug)]
pub struct Child_Data {
    tx_channel: Sender<ProtocolMessage>,
    rx_channel: Receiver<ProtocolMessage>,
    name: String,
    num_actions: u32,
}

/// Coordinator
/// Struct maintaining state for coordinator
#[derive(Debug)]
pub struct Coordinator {
    state: CoordinatorState,
    running: Arc<AtomicBool>,
    log: oplog::OpLog,
    num_clients: u32,
    num_requests: u32,
    num_participants: u32,
    clients: Vec<Child_Data>,
    participants: Vec<Child_Data>,
    successful_ops: u64,
    failed_ops: u64,
    unknown_ops: u64,

}

///
/// Coordinator
/// Implementation of coordinator functionality
/// Required:
/// 1. new -- Constructor
/// 2. protocol -- Implementation of coordinator side of protocol
/// 3. report_status -- Report of aggregate commit/abort/unknown stats on exit.
/// 4. participant_join -- What to do when a participant joins
/// 5. client_join -- What to do when a client joins
///
impl Coordinator {

    ///
    /// new()
    /// Initialize a new coordinator
    ///
    /// <params>
    ///     log_path: directory for log files --> create a new log there.
    ///     r: atomic bool --> still running?
    ///
    pub fn new(
        log_path: String,
        r: &Arc<AtomicBool>,
        opts: &tpcoptions::TPCOptions) -> Coordinator {

        Coordinator {
            state: CoordinatorState::Quiescent,
            log: oplog::OpLog::new(log_path),
            running: r.clone(),
            // TODO
            num_clients: opts.num_clients,
            num_requests: opts.num_requests,
            num_participants: opts.num_participants,
            clients: vec![],
            participants: vec![],
            successful_ops: 0,
            failed_ops: 0,
            unknown_ops: 0,
        }
    }

    ///
    /// participant_join()
    /// Adds a new participant for the coordinator to keep track of
    ///
    /// HINT: Keep track of any channels involved!
    /// HINT: You may need to change the signature of this function
    ///
    pub fn participant_join(&mut self, name: &String, tx_channel: Sender<ProtocolMessage>, rx_channel: Receiver<ProtocolMessage>) {
        assert!(self.state == CoordinatorState::Quiescent);
        
        // TODO
        let participant: Child_Data = Child_Data {
            tx_channel: tx_channel,
            rx_channel: rx_channel,
            name: name.clone(),
            num_actions: 0,
        };
        self.participants.push(participant);
    }

    ///
    /// client_join()
    /// Adds a new client for the coordinator to keep track of
    ///
    /// HINT: Keep track of any channels involved!
    /// HINT: You may need to change the signature of this function
    ///
    pub fn client_join(&mut self, name: &String, tx_channel: Sender<ProtocolMessage>, rx_channel: Receiver<ProtocolMessage>) {
        assert!(self.state == CoordinatorState::Quiescent);

        // TODO
        let client: Child_Data = Child_Data {
            tx_channel: tx_channel,
            rx_channel: rx_channel,
            name: name.clone(),
            num_actions: 0,
        };
        self.clients.push(client);
    }

    ///
    /// report_status()
    /// Report the abort/commit/unknown status (aggregate) of all transaction
    /// requests made by this coordinator before exiting.
    ///
    pub fn report_status(&mut self) {
        // TODO: Collect actual stats
        //let successful_ops: u64 = 0;
        //let failed_ops: u64 = 0;
        //let unknown_ops: u64 = 0;

        println!("coordinator     :\tCommitted: {:6}\tAborted: {:6}\tUnknown: {:6}", self.successful_ops, self.failed_ops, self.unknown_ops);
    }

    ///
    /// protocol()
    /// Implements the coordinator side of the 2PC protocol
    /// HINT: If the simulation ends early, don't keep handling requests!
    /// HINT: Wait for some kind of exit signal before returning from the protocol!
    ///
    pub fn protocol(&mut self) {

        // TODO
        loop {
            if !self.running.load(Ordering::SeqCst) {
                break;
            }
            thread::sleep(Duration::from_millis(100));
        }
        self.report_status();
    }
}
