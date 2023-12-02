//!
//! participant.rs
//! Implementation of 2PC participant
//!
extern crate ipc_channel;
extern crate log;
extern crate rand;
extern crate stderrlog;

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use std::thread;

use participant::rand::prelude::*;
use participant::ipc_channel::ipc::IpcReceiver as Receiver;
use participant::ipc_channel::ipc::TryRecvError;
use participant::ipc_channel::ipc::IpcSender as Sender;

use message::MessageType;
use message::ProtocolMessage;
use message::RequestStatus;
use oplog;

///
/// ParticipantState
/// enum for Participant 2PC state machine
///
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParticipantState {
    Quiescent,
    ReceivedP1,
    VotedAbort,
    VotedCommit,
    AwaitingGlobalDecision,
}

///
/// Participant
/// Structure for maintaining per-participant state and communication/synchronization objects to/from coordinator
///
#[derive(Debug)]
pub struct Participant {
    id_str: String,
    state: ParticipantState,
    log: oplog::OpLog,
    running: Arc<AtomicBool>,
    send_success_prob: f64,
    operation_success_prob: f64,
    server_name: String,
    num_requests: u32,
    total_requests: u32,
    failed_ops: u64,
    successful_ops: u64,
    unknown_ops: u64,
    tx_channel: Sender<ProtocolMessage>,
    rx_channel: Receiver<ProtocolMessage>,
}

///
/// Participant
/// Implementation of participant for the 2PC protocol
/// Required:
/// 1. new -- Constructor
/// 2. pub fn report_status -- Reports number of committed/aborted/unknown for each participant
/// 3. pub fn protocol() -- Implements participant side protocol for 2PC
///
impl Participant {

    ///
    /// new()
    ///
    /// Return a new participant, ready to run the 2PC protocol with the coordinator.
    ///
    /// HINT: You may want to pass some channels or other communication
    ///       objects that enable coordinator->participant and participant->coordinator
    ///       messaging to this constructor.
    /// HINT: You may want to pass some global flags that indicate whether
    ///       the protocol is still running to this constructor. There are other
    ///       ways to communicate this, of course.
    ///
    pub fn new(
        id_str: String,
        log_path: String,
        r: &Arc<AtomicBool>,
        send_success_prob: f64,
        operation_success_prob: f64,
        server_name: String,
        num_requests: u32,
        total_requests: u32,
        tx: Sender<ProtocolMessage>,
        rx: Receiver<ProtocolMessage>) -> Participant {

        Participant {
            id_str: id_str,
            state: ParticipantState::Quiescent,
            log: oplog::OpLog::new(log_path),
            running: r.clone(),
            send_success_prob: send_success_prob,
            operation_success_prob: operation_success_prob,
            // TODO
            server_name: server_name,
            num_requests: num_requests,
            total_requests: total_requests,
            successful_ops: 0,
            failed_ops: 0,
            unknown_ops: 0,
            tx_channel: tx,
            rx_channel: rx,
        }
    }

    ///
    /// send()
    /// Send a protocol message to the coordinator. This can fail depending on
    /// the success probability. For testing purposes, make sure to not specify
    /// the -S flag so the default value of 1 is used for failproof sending.
    ///
    /// HINT: You will need to implement the actual sending
    ///
    pub fn send(&mut self, pm: ProtocolMessage) {
        let x: f64 = random();
        // FOR NOW ALWAYS SEND TO MAKE SURE STUFF FINISHES
        
        if x <= self.send_success_prob && self.running.load(Ordering::SeqCst) {
            // TODO: Send success
            match self.tx_channel.send(pm) {
                Ok(_) => {

                }, Err(_) => {

                }
            }

        } else {
            // TODO: Send fail
        }
    }

    ///
    /// perform_operation
    /// Perform the operation specified in the 2PC proposal,
    /// with some probability of success/failure determined by the
    /// command-line option success_probability.
    ///
    /// HINT: The code provided here is not complete--it provides some
    ///       tracing infrastructure and the probability logic.
    ///       Your implementation need not preserve the method signature
    ///       (it's ok to add parameters or return something other than
    ///       bool if it's more convenient for your design).
    ///
    pub fn perform_operation(&mut self, request_option: &Option<ProtocolMessage>) -> bool {

        trace!("{}::Performing operation", self.id_str.clone());
        let x: f64 = random();
        if request_option.is_some() {
            self.unknown_ops += 1;
            let pm: ProtocolMessage = request_option.as_ref().unwrap().clone();
            if x <= self.operation_success_prob {
                // TODO: Successful operation
                let success_pm = ProtocolMessage::generate(
                    MessageType::ParticipantVoteCommit, 
                    pm.txid.clone(), 
                    pm.senderid.clone(),
                    pm.opid.clone());
                self.send(success_pm);
                // Log the participant vote success
                self.log.append(MessageType::ParticipantVoteCommit, pm.txid.clone(), pm.senderid.clone(), pm.opid.clone());
                //println!("{} sent success pm, senderid: {}",self.id_str.clone(),pm.senderid.clone());

            } else {
                // TODO: Failed operation
                let fail_pm = ProtocolMessage::generate(
                    MessageType::ParticipantVoteAbort, 
                    pm.txid.clone(), 
                    pm.senderid.clone(),
                    pm.opid.clone());
                self.send(fail_pm);
                // Log the participant vote fail
                self.log.append(MessageType::ParticipantVoteAbort,pm.txid.clone(),pm.senderid.clone(),pm.opid.clone());
                //println!("{} sent success pm, senderid: {}",self.id_str.clone(),pm.senderid.clone());
            }
            if self.running.load(Ordering::SeqCst) {
                match self.rx_channel.recv() {
                    Ok(coordinator_result) => {
                        self.log.append(coordinator_result.mtype,coordinator_result.txid,coordinator_result.senderid,coordinator_result.opid);
                        self.unknown_ops -= 1;
                        if coordinator_result.mtype == MessageType::CoordinatorCommit {
                            // Log the coordinator success
                            self.successful_ops += 1;
                        }
                        else if coordinator_result.mtype == MessageType::CoordinatorExit {
                            // Log the coordinator fail
                            self.failed_ops += 1;
                            return false;
                        }
                        else {
                            self.failed_ops += 1;
                        }
                    }, Err(_) => {
                        self.failed_ops += 1;
                    }
                } 

            }
        }

        true
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

        println!("{:16}:\tCommitted: {:6}\tAborted: {:6}\tUnknown: {:6}", self.id_str.clone(), self.successful_ops, self.failed_ops, self.unknown_ops);
    }

    ///
    /// wait_for_exit_signal(&mut self)
    /// Wait until the running flag is set by the CTRL-C handler
    ///
    pub fn wait_for_exit_signal(&mut self) {
        trace!("{}::Waiting for exit signal", self.id_str.clone());

        // TODO
        if self.running.load(Ordering::SeqCst) {
            let _ = self.rx_channel.recv().unwrap();
        }
        /*
        while self.running.load(Ordering::SeqCst) {
            thread::sleep(Duration::from_millis(100));
        }
        */
        trace!("{}::Exiting", self.id_str.clone());
    }

    ///
    /// protocol()
    /// Implements the participant side of the 2PC protocol
    /// HINT: If the simulation ends early, don't keep handling requests!
    /// HINT: Wait for some kind of exit signal before returning from the protocol!
    ///
    pub fn protocol(&mut self) {
        trace!("{}::Beginning protocol", self.id_str.clone());

        // TODO:
        while self.failed_ops+self.successful_ops < self.total_requests as u64 && self.running.load(Ordering::SeqCst) {
            match self.rx_channel.recv(){
                Ok(request_option) => {
                    self.log.append(request_option.mtype.clone(), request_option.txid.clone(), request_option.senderid.clone(), request_option.opid.clone());
        
                    self.perform_operation(&Some(request_option));

                }, Err(_) => {
                    break;
                }
            }
        }

        self.wait_for_exit_signal();
        self.report_status();
    }
}