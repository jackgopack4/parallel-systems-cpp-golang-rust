//!
//! coordinator.rs
//! Implementation of 2PC coordinator
//!
extern crate log;
extern crate stderrlog;
extern crate rand;
extern crate ipc_channel;
extern crate queues;

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;
use std::time::Instant;

use coordinator::ipc_channel::ipc::IpcSender as Sender;
use coordinator::ipc_channel::ipc::IpcReceiver as Receiver;
use coordinator::ipc_channel::ipc::TryRecvError;
use coordinator::ipc_channel::ipc::channel;
use coordinator::queues::*;

use message;
use message::MessageType;
use message::ProtocolMessage;
use message::RequestStatus;
use oplog;
use tpcoptions;
//use queues::*;

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
    pub tx_channel: Sender<ProtocolMessage>,
    pub rx_channel: Receiver<ProtocolMessage>,
    pub name: String,
    pub num_actions: u32,
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
    total_requests: u64,
    clients: Vec<Child_Data>,
    participants: Vec<Child_Data>,
    successful_ops: u64,
    failed_ops: u64,
    unknown_ops: u64,
    server_name: String,
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
        opts: &tpcoptions::TPCOptions,
        server_name: &String) -> Coordinator {
        let total_num:u64 = (opts.num_clients*opts.num_requests) as u64;
        Coordinator {
            state: CoordinatorState::Quiescent,
            log: oplog::OpLog::new(log_path),
            running: r.clone(),
            // TODO
            num_clients: opts.num_clients,
            num_requests: opts.num_requests,
            num_participants: opts.num_participants,
            total_requests: total_num,
            clients: vec![],
            participants: vec![],
            successful_ops: 0,
            failed_ops: 0,
            unknown_ops: 0,
            server_name: server_name.clone(),
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

        println!("coordinator     :\tCommitted: {:6}\tAborted: {:6}\tUnknown: {:6}", 
                    self.successful_ops, self.failed_ops, self.unknown_ops);
    }

    ///
    /// protocol()
    /// Implements the coordinator side of the 2PC protocol
    /// HINT: If the simulation ends early, don't keep handling requests!
    /// HINT: Wait for some kind of exit signal before returning from the protocol!
    ///
    pub fn protocol(&mut self) {
        //println!("running coord protocol total_requests {}",self.total_requests);
        // TODO
        let mut client_q: Queue<ProtocolMessage> = queue![];
        let mut cur_rqst: ProtocolMessage = ProtocolMessage::generate(MessageType::ClientRequest, format!("dummy"), format!("none"), 0);
        let mut client_rx_idx:usize = 0;
        while self.successful_ops+self.failed_ops < self.total_requests && self.running.load(Ordering::SeqCst) {
            match self.state {
                CoordinatorState::Quiescent => { 
                    while client_q.size() == 0 && self.running.load(Ordering::SeqCst){
                        let mut idx: usize = 0;
                        while idx < self.num_clients as usize && self.running.load(Ordering::SeqCst) {
                            match self.clients[client_rx_idx].rx_channel.try_recv() {
                                Ok(res) => {
                                    // Do something interesting with your result
                                    //println!("Received data from client {}",idx);
                                    self.log.append(res.mtype.clone(), res.txid.clone(), res.senderid.clone(), res.opid.clone());
                                    client_q.add(res).unwrap();
                                    self.unknown_ops += 1;
                                    //break;
                                },
                                Err(_) => {
                                    // Do something else useful while we wait
                                    //println!("Incrementing counter to check next client for traffic");
                                }
                            }
                            idx += 1;
                            client_rx_idx += 1;
                            if client_rx_idx == self.num_clients as usize {
                                client_rx_idx = 0;
                            }
                        }
                        client_rx_idx += 1;
                        if client_rx_idx == self.num_clients as usize {
                            client_rx_idx = 0;
                        }
                        //println!("coord broke out of rx loop");
                        
                    }
                    /*
                    if !self.running.load(Ordering::SeqCst) {
                        break;
                    }
                    */
                    cur_rqst = client_q.remove().unwrap().clone();
                    self.state = CoordinatorState::ReceivedRequest;
                },
                CoordinatorState::ReceivedRequest => {
                    /*
                    if !self.running.load(Ordering::SeqCst) {
                        break;
                    }
                    */
                    let mut tmp_rqst: ProtocolMessage;
                    let mut i: usize = 0;
                    while i < self.num_participants as usize && self.running.load(Ordering::SeqCst){
                        //let mut child_data: &Child_Data = &self.participants[i];
                        tmp_rqst = ProtocolMessage::generate(
                            MessageType::CoordinatorPropose, 
                            cur_rqst.txid.clone(), 
                            cur_rqst.senderid.clone(), 
                            cur_rqst.opid.clone());
                        match self.participants[i].tx_channel.send(tmp_rqst.clone()) {
                            Ok(_) => {

                            }, Err(_) => {
                                i = self.num_participants as usize;
                            }
                        }
                        i+=1;
                        //println!("sent request to participant {}",i);
                    }
                    self.log.append(MessageType::CoordinatorPropose, cur_rqst.txid.clone(), cur_rqst.senderid.clone(), cur_rqst.opid.clone());

                    //println!("sent request to all {} participants",self.num_participants);
                    //println!("coord succesful: {}, failed: {}, unknown: {}, total: {}",self.successful_ops,self.failed_ops,self.failed_ops,self.total_requests);
                    self.state = CoordinatorState::ProposalSent;
                },
                CoordinatorState::ProposalSent => {
                    /* 
                    if !self.running.load(Ordering::SeqCst) {
                        break;
                    }
                    */
                    let mut count_received: u32 = 0;
                    let mut count_fail: u32 = 0;
                    //println!("starting loop to receive from participants");
                    let start_time = Instant::now();
                    let mut time_passed = Instant::now().duration_since(start_time);
                    //let factor: f32 = 1.06;
                    let timeout_val: u128 = std::cmp::max(5*self.num_participants*self.num_clients, 10) as u128;
                    while count_received < self.num_participants && time_passed.as_millis() < timeout_val && self.running.load(Ordering::SeqCst) {
                        let mut i:usize = 0;
                        while i < self.num_participants as usize && self.running.load(Ordering::SeqCst) {
                            //let participant_index: usize = i as usize;
                            match self.participants[i].rx_channel.try_recv() {
                                Ok(res) => {
                                    // Do something interesting with your result
                                    //println!("Received data from participant {}",i);
                                    if res.mtype == MessageType::ParticipantVoteAbort {
                                        count_fail += 1;
                                    }
                                    count_received += 1;
                                    /*
                                    if i == 0 {
                                        cur_rqst = res.clone();
                                    }
                                    */
                                    //println!("count received from participants: {}",count_received);
                                    //client_q.add(res);
                                    //break;
                                },
                                Err(_) => {
                                    // Do something else useful while we wait
                                    //thread::sleep(Duration::from_micros(10));
                                    //println!("no message yet, moving to next participant");
                                }
                            }
                            i += 1;
                            //println!("coordinator currently received {}",count_received);
                        }
                        time_passed = Instant::now().duration_since(start_time)
                    }
                    if count_fail > 0 || count_received < self.num_participants {
                        self.state = CoordinatorState::ReceivedVotesAbort;
                        //println!("coordinator moving to abort");
                    } else {
                        self.state = CoordinatorState::ReceivedVotesCommit;
                        //println!("coordinator moving to commit");
                    }
                },
                CoordinatorState::ReceivedVotesAbort => {
                    /* 
                    if !self.running.load(Ordering::SeqCst) {
                        break;
                    }
                    */
                    let mut tmp_abort_msg: ProtocolMessage;
                    let mut i:usize = 0;
                    while i < self.num_participants as usize && self.running.load(Ordering::SeqCst){
                        tmp_abort_msg = ProtocolMessage::generate(
                            MessageType::CoordinatorAbort, 
                            cur_rqst.txid.clone(), 
                            cur_rqst.senderid.clone(), 
                            cur_rqst.opid.clone());
                        self.participants[i].tx_channel.send(tmp_abort_msg.clone()).unwrap();
                        i += 1;
                    }
                    i = 0;
                    while i < self.num_clients as usize && self.running.load(Ordering::SeqCst) {
                        if self.clients[i].name == cur_rqst.senderid {
                            tmp_abort_msg = ProtocolMessage::generate(
                                MessageType::CoordinatorAbort, 
                                cur_rqst.txid.clone(), 
                                cur_rqst.senderid.clone(), 
                                cur_rqst.opid.clone());
                            self.clients[i].tx_channel.send(tmp_abort_msg.clone()).unwrap();
                        }
                        i += 1;
                    }
                    self.log.append(MessageType::CoordinatorAbort,cur_rqst.txid.clone(),cur_rqst.senderid.clone(),cur_rqst.opid.clone());
                    self.failed_ops += 1;
                    self.unknown_ops -= 1;
                    self.state = CoordinatorState::SentGlobalDecision;
                },
                CoordinatorState::ReceivedVotesCommit => {
                    /* 
                    if !self.running.load(Ordering::SeqCst) {
                        break;
                    }
                    */
                    let mut tmp_commit_msg: ProtocolMessage; 
                    let mut i:usize = 0;
                    while i < self.num_participants as usize && self.running.load(Ordering::SeqCst) {
                        tmp_commit_msg = ProtocolMessage::generate(
                            MessageType::CoordinatorCommit, 
                            cur_rqst.txid.clone(), 
                            cur_rqst.senderid.clone(), 
                            cur_rqst.opid.clone());
                        self.participants[i].tx_channel.send(tmp_commit_msg.clone()).unwrap();
                        i += 1;
                    }
                    i = 0;
                    while i < self.num_clients as usize && self.running.load(Ordering::SeqCst){
                        if self.clients[i].name == cur_rqst.senderid {
                            tmp_commit_msg = ProtocolMessage::generate(
                                MessageType::CoordinatorCommit, 
                                cur_rqst.txid.clone(), 
                                cur_rqst.senderid.clone(), 
                                cur_rqst.opid.clone());
                            self.clients[i].tx_channel.send(tmp_commit_msg.clone()).unwrap();
                        }
                        i += 1;
                    }
                    self.log.append(MessageType::CoordinatorCommit,cur_rqst.txid.clone(),cur_rqst.senderid.clone(),cur_rqst.opid.clone());

                    self.successful_ops += 1;
                    self.unknown_ops -= 1;
                    self.state = CoordinatorState::SentGlobalDecision;
                },
                CoordinatorState::SentGlobalDecision => {
                    self.state = CoordinatorState::Quiescent;
                    if self.successful_ops+self.failed_ops == self.total_requests {
                        break;
                    }
                },
            }
        }
        if self.running.load(Ordering::SeqCst) {
            println!("coord finished, exiting now!");
        } else {
            println!("coord aborted, wrapping things up.")
        }
        /*
        while self.running.load(Ordering::SeqCst) {
            thread::sleep(Duration::from_millis(10));
        }
        */
        let mut exit_msg: ProtocolMessage;
        for i in 0..self.num_participants as usize {
            exit_msg = ProtocolMessage::generate(
                MessageType::CoordinatorExit, 
                format!("coordinator_op_1"), 
                format!("coordinator"), 
                1);
            match self.participants[i].tx_channel.send(exit_msg.clone()) {
                Ok(_) => {

                }, Err(_) => {

                }
            }
        }
        for i in 0..self.num_clients as usize {
            exit_msg = ProtocolMessage::generate(
                MessageType::CoordinatorExit, 
                format!("coordinator_op_1"), 
                format!("coordinator"), 
                1);
            match self.clients[i].tx_channel.send(exit_msg.clone()) {
                Ok(_) => {

                }, Err(_) => {

                }
            }
        }
        self.report_status();
    }
}
