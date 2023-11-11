//!
//! client.rs
//! Implementation of 2PC client
//!
extern crate ipc_channel;
extern crate log;
extern crate stderrlog;

use std::thread;
use std::time::Duration;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::collections::HashMap;

use client::ipc_channel::ipc::IpcReceiver as Receiver;
use client::ipc_channel::ipc::TryRecvError;
use client::ipc_channel::ipc::IpcSender as Sender;

use message;
use message::MessageType;
use message::RequestStatus;

use crate::message::ProtocolMessage;

// Client state and primitives for communicating with the coordinator
#[derive(Debug)]
pub struct Client {
    pub id_str: String,
    pub running: Arc<AtomicBool>,
    pub num_requests: u32,
    pub server_name: String,
    pub tx_channel: Sender<ProtocolMessage>,
    pub rx_channel: Receiver<ProtocolMessage>,
    pub successful_ops: u32,
    pub failed_ops: u32,
    pub unknown_ops: u32,
    pub cur_rqst: u32
}

///
/// Client Implementation
/// Required:
/// 1. new -- constructor
/// 2. pub fn report_status -- Reports number of committed/aborted/unknown
/// 3. pub fn protocol(&mut self, n_requests: i32) -- Implements client side protocol
///
impl Client {

    ///
    /// new()
    ///
    /// Constructs and returns a new client, ready to run the 2PC protocol
    /// with the coordinator.
    ///
    /// HINT: You may want to pass some channels or other communication
    ///       objects that enable coordinator->client and client->coordinator
    ///       messaging to this constructor.
    /// HINT: You may want to pass some global flags that indicate whether
    ///       the protocol is still running to this constructor
    ///
    pub fn new(id_str: String,
               running: &Arc<AtomicBool>,
               num_requests: u32,
               server_name: &String,
               tx: Sender<ProtocolMessage>,
               rx: Receiver<ProtocolMessage>) -> Client {
        Client {
            id_str: id_str,
            running: running.clone(),
            num_requests: num_requests,
            // TODO
            server_name: server_name.clone(),
            tx_channel: tx,
            rx_channel: rx,
            successful_ops: 0,
            failed_ops: 0,
            unknown_ops: 0,
            cur_rqst: 0,
        }
    }

    ///
    /// wait_for_exit_signal(&mut self)
    /// Wait until the running flag is set false by the CTRL-C handler
    ///
    pub fn wait_for_exit_signal(&mut self) {
        trace!("{}::Waiting for exit signal", self.id_str.clone());

        // TODO
        let _ = self.rx_channel.recv().unwrap();
        /*
        while self.running.load(Ordering::SeqCst) {
            thread::sleep(Duration::from_millis(100));
        }
        */
        trace!("{}::Exiting", self.id_str.clone());
    }

    ///
    /// send_next_operation(&mut self)
    /// Send the next operation to the coordinator
    ///
    pub fn send_next_operation(&mut self) {

        // Create a new request with a unique TXID.
        //self.num_requests = self.num_requests + 1;
        let txid = format!("{}_op_{}", self.id_str.clone(), self.cur_rqst);
        let pm = message::ProtocolMessage::generate(message::MessageType::ClientRequest,
                                                    txid.clone(),
                                                    self.id_str.clone(),
                                                    self.cur_rqst);
        info!("{}::Sending operation #{}", self.id_str.clone(), self.num_requests);
        // TODO
        self.tx_channel.send(pm).unwrap();
        self.unknown_ops += 1;

        self.cur_rqst += 1;
        trace!("{}::Sent operation #{}", self.id_str.clone(), self.num_requests);
    }

    ///
    /// recv_result()
    /// Wait for the coordinator to respond with the result for the
    /// last issued request. Note that we assume the coordinator does
    /// not fail in this simulation
    ///
    pub fn recv_result(&mut self) {

        info!("{}::Receiving Coordinator Result", self.id_str.clone());

        // TODO
        if self.successful_ops >= self.num_requests {
            return;
        }
        let rx_msg = self.rx_channel.recv().unwrap();
        self.unknown_ops -= 1;
        if rx_msg.mtype == MessageType::CoordinatorAbort {
            self.failed_ops += 1;
        } else if rx_msg.mtype == MessageType::CoordinatorCommit {
            self.successful_ops += 1;
        }
        //println!("client successful ops: {}",self.successful_ops);
    }

    ///
    /// report_status()
    /// Report the abort/commit/unknown status (aggregate) of all transaction
    /// requests made by this client before exiting.
    ///
    pub fn report_status(&mut self) {
        // TODO: Collect actual stats
        /* 
        let successful_ops: u64 = 0;
        let failed_ops: u64 = 0;
        let unknown_ops: u64 = 0;
        */
        println!("{:16}:\tCommitted: {:6}\tAborted: {:6}\tUnknown: {:6}", self.id_str.clone(), self.successful_ops, self.failed_ops, self.unknown_ops);
    }

    ///
    /// protocol()
    /// Implements the client side of the 2PC protocol
    /// HINT: if the simulation ends early, don't keep issuing requests!
    /// HINT: if you've issued all your requests, wait for some kind of
    ///       exit signal before returning from the protocol method!
    ///
    pub fn protocol(&mut self, n_requests: u32) {

        // TODO
        //println!("{}",format!("running client protocol child id {}, num_requests {}",self.id_str, self.num_requests));
        while ((self.failed_ops+self.successful_ops) < self.num_requests) {
            //println!("{} starting next round of client tx/rx, num_requests: {}, failed: {}, success: {}",self.id_str,self.num_requests,self.failed_ops,self.successful_ops);
            self.send_next_operation();
            /* 
            if !self.running.load(Ordering::SeqCst) {
                break;
            }
            */
            self.recv_result();
            //println!("successful ops at client in protocol: {}",self.successful_ops);
            //println!("failed ops at client in protocol: {}",self.failed_ops);
        }
        self.wait_for_exit_signal();
        self.report_status();
    }
}
