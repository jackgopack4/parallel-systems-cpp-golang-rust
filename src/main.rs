#[macro_use]
extern crate log;
extern crate stderrlog;
extern crate clap;
extern crate ctrlc;
extern crate ipc_channel;
use std::env;
use std::fs;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::process::{Child,Command};
use ipc_channel::ipc::IpcSender as Sender;
use ipc_channel::ipc::IpcReceiver as Receiver;
use ipc_channel::ipc::IpcOneShotServer;
use ipc_channel::ipc::channel;
pub mod message;
pub mod oplog;
pub mod coordinator;
pub mod participant;
pub mod client;
pub mod checker;
pub mod tpcoptions;
use message::ProtocolMessage;

use crate::coordinator::Child_Data;
use crate::tpcoptions::TPCOptions;

///
/// pub fn spawn_child_and_connect(child_opts: &mut tpcoptions::TPCOptions) -> (std::process::Child, Sender<ProtocolMessage>, Receiver<ProtocolMessage>)
///
///     child_opts: CLI options for child process
///
/// 1. Set up IPC
/// 2. Spawn a child process using the child CLI options
/// 3. Do any required communication to set up the parent / child communication channels
/// 4. Return the child process handle and the communication channels for the parent
///
/// HINT: You can change the signature of the function if necessary
///
fn spawn_child_and_connect(child_opts: &mut tpcoptions::TPCOptions, server:IpcOneShotServer::<(Sender<ProtocolMessage>,Receiver<ProtocolMessage>)>) -> (Child, Sender<ProtocolMessage>, Receiver<ProtocolMessage>) {
    let child = Command::new(env::current_exe().unwrap())
        .args(child_opts.as_vec())
        .spawn()
        .expect("Failed to execute child process");
    println!("{}",format!("spawned child pid: {}, id: {}",child.id(),child_opts.num));
    let (_,(tx_to_child,rx_from_child)) = server.accept().unwrap();
    //let (tx, rx) = channel().unwrap();
    // TODO
    //let tx0 = Sender::connect(child_opts.ipc_path.clone()).unwrap();
    //tx0.send(tx).unwrap();
    println!("received tx,rx for child id: {}",child_opts.num);
    (child, tx_to_child, rx_from_child)
}

///
/// pub fn connect_to_coordinator(opts: &tpcoptions::TPCOptions) -> (Sender<ProtocolMessage>, Receiver<ProtocolMessage>)
///
///     opts: CLI options for this process
///
/// 1. Connect to the parent via IPC
/// 2. Do any required communication to set up the parent / child communication channels
/// 3. Return the communication channels for the child
///
/// HINT: You can change the signature of the function if necessasry
///
fn connect_to_coordinator(opts: &tpcoptions::TPCOptions) -> (Sender<ProtocolMessage>, Receiver<ProtocolMessage>) {
    let (tx_to_parent, rx_from_child) = channel().unwrap();
    let (tx_to_child, rx_from_parent) = channel().unwrap();
    let server_name = opts.ipc_path.clone();
    println!("{}",format!("connecting to coordinator, id: {}",opts.num));
    let tx0: Sender<(Sender<ProtocolMessage>,Receiver<ProtocolMessage>)> = Sender::connect(server_name).unwrap();
    println!("{}",format!("created server tx channel, id: {}",opts.num));
    // TODO
    tx0.send((tx_to_child,rx_from_child)).unwrap();
    println!("{}",format!("sent child tx/rx channels, id: {}",opts.num));
    (tx_to_parent, rx_from_parent)
}

///
/// pub fn run(opts: &tpcoptions:TPCOptions, running: Arc<AtomicBool>)
///     opts: An options structure containing the CLI arguments
///     running: An atomically reference counted (ARC) AtomicBool(ean) that is
///         set to be false whenever Ctrl+C is pressed
///
/// 1. Creates a new coordinator
/// 2. Spawns and connects to new clients processes and then registers them with
///    the coordinator
/// 3. Spawns and connects to new participant processes and then registers them
///    with the coordinator
/// 4. Starts the coordinator protocol
/// 5. Wait until the children finish execution
///
fn run(opts: &tpcoptions::TPCOptions, running: Arc<AtomicBool>) {
    let coord_log_path = format!("{}//{}", opts.log_path, "coordinator.log");
    // TODO
    println!("running, send_prob:{}, op_prob:{}, clients:{}, requests:{}, participants:{}",
                opts.send_success_probability,
                opts.operation_success_probability,
                opts.num_clients,
                opts.num_requests,
                opts.num_participants);
    // starting with one client and one participant, one request each
    //opts.ipc_path = server_name.clone();

    let (_, server_name) = IpcOneShotServer::<(Sender<ProtocolMessage>,Receiver<ProtocolMessage>)>::new().unwrap();
    let mut coord: coordinator::Coordinator = coordinator::Coordinator::new(
                coord_log_path,
                &running,
                opts,
                &server_name,
            );
    //let mut server_vec = vec![];
    //let mut name_vec = vec![];
    // client_vec is tuple of (child,sender,receiver)
    //let mut client_vec: Vec<(Child, Sender<ProtocolMessage>, Receiver<ProtocolMessage>)> = vec![];
    //let mut participant_vec: Vec<(Child, Sender<ProtocolMessage>, Receiver<ProtocolMessage>)> = vec![];
    let mut participant_child_vec: Vec<Child_Data> = vec![];
    /* 
    for _ in 0..opts.num_clients {
        let (tmp_server, tmp_name) = IpcOneShotServer::<(Sender<ProtocolMessage>,Receiver<ProtocolMessage>)>::new().unwrap();
        server_vec.push(tmp_server);
        name_vec.push(tmp_name);
    }
    */
    for i in 0..opts.num_clients {
        //let vec_index:usize = i as usize;
        let (server, server_name) = IpcOneShotServer::<(Sender<ProtocolMessage>,Receiver<ProtocolMessage>)>::new().unwrap();
        let mut client_opts: TPCOptions =  TPCOptions{
            send_success_probability: opts.send_success_probability.clone(),
            operation_success_probability: opts.operation_success_probability.clone(),
            num_clients: opts.num_clients.clone(),
            num_participants: opts.num_participants.clone(),
            num_requests: opts.num_requests.clone(),
            verbosity: opts.verbosity.clone(),
            mode: "client".to_string(),
            log_path: opts.log_path.clone(),
            ipc_path: server_name.clone(),
            num: i,
        };
        let (_temp_client, temp_tx, temp_rx) = spawn_child_and_connect(&mut client_opts,server);
        let temp_name: String = format!("client_{}",i);
        coord.client_join(&temp_name, temp_tx, temp_rx);
        //client_vec.push(spawn_child_and_connect(&mut client_opts,server));
        //let (_,(tx_to_child,tx_from_child)) = server.accept().unwrap();
    }

    for i in 0..opts.num_participants {
        let (server,server_name) = IpcOneShotServer::<(Sender<ProtocolMessage>,Receiver<ProtocolMessage>)>::new().unwrap();
        let mut participant_opts: TPCOptions =  TPCOptions{
            send_success_probability: opts.send_success_probability.clone(),
            operation_success_probability: opts.operation_success_probability.clone(),
            num_clients: opts.num_clients.clone(),
            num_participants: opts.num_participants.clone(),
            num_requests: opts.num_requests.clone(),
            verbosity: opts.verbosity.clone(),
            mode: "participant".to_string(),
            log_path: opts.log_path.clone(),
            ipc_path: server_name.clone(),
            num: i,
        };
        let (_temp_participant, temp_tx, temp_rx) = spawn_child_and_connect(&mut participant_opts,server);
        let temp_name: String = format!("participant_{}",i);
        /* 
        participant_child_vec.push(Child_Data{
            tx_channel:temp_tx,
            rx_channel:temp_rx,
            name:format!("participant_{}",i),
            num_actions:0,
        });
        */
        coord.participant_join(&temp_name,temp_tx,temp_rx);
        //participant_vec.push(spawn_child_and_connect(&mut participant_opts, server));
    }

    coord.protocol();

}

///
/// pub fn run_client(opts: &tpcoptions:TPCOptions, running: Arc<AtomicBool>)
///     opts: An options structure containing the CLI arguments
///     running: An atomically reference counted (ARC) AtomicBool(ean) that is
///         set to be false whenever Ctrl+C is pressed
///
/// 1. Connects to the coordinator to get tx/rx
/// 2. Constructs a new client
/// 3. Starts the client protocol
///
fn run_client(opts: &tpcoptions::TPCOptions, running: Arc<AtomicBool>) {
    // TODO
    println!("running client {}, send_prob:{}, op_prob:{}, clients:{}, requests:{}, participants:{}",
                opts.num,
                opts.send_success_probability,
                opts.operation_success_probability,
                opts.num_clients,
                opts.num_requests,
                opts.num_participants);
    let client_id_str: String = format!("client_{}", opts.num);
    let (tx,rx) = connect_to_coordinator(opts); 
    let mut client: client::Client = client::Client::new(
        client_id_str,
        &running,
        opts.num_requests.clone(),
        &opts.ipc_path,
        tx,
        rx
    );
    client.protocol(0);           
}

///
/// pub fn run_participant(opts: &tpcoptions:TPCOptions, running: Arc<AtomicBool>)
///     opts: An options structure containing the CLI arguments
///     running: An atomically reference counted (ARC) AtomicBool(ean) that is
///         set to be false whenever Ctrl+C is pressed
///
/// 1. Connects to the coordinator to get tx/rx
/// 2. Constructs a new participant
/// 3. Starts the participant protocol
///
fn run_participant(opts: & tpcoptions::TPCOptions, running: Arc<AtomicBool>) {
    let participant_id_str: String = format!("participant_{}", opts.num);
    let participant_log_path: String = format!("{}//{}.log", opts.log_path, participant_id_str);
    println!("running participant {}, send_prob:{}, op_prob:{}, clients:{}, requests:{}, participants:{}",
                opts.num,
                opts.send_success_probability,
                opts.operation_success_probability,
                opts.num_clients,
                opts.num_requests,
                opts.num_participants);
    // TODO
    let (tx,rx) = connect_to_coordinator(opts); 

    let mut participant: participant::Participant = participant::Participant::new(
        participant_id_str,
        participant_log_path,
        &running,
        opts.send_success_probability.clone(),
        opts.operation_success_probability.clone(),
        opts.ipc_path.clone(),
        opts.num_requests.clone(),
        (opts.num_requests*opts.num_clients).clone(),
        tx,
        rx
    );
    participant.protocol();
}

fn main() {
    // Parse CLI arguments
    let opts: TPCOptions = tpcoptions::TPCOptions::new();
    // Set-up logging and create OpLog path if necessary
    stderrlog::new()
            .module(module_path!())
            .quiet(false)
            .timestamp(stderrlog::Timestamp::Millisecond)
            .verbosity(opts.verbosity)
            .init()
            .unwrap();
    match fs::create_dir_all(opts.log_path.clone()) {
        Err(e) => error!("Failed to create log_path: \"{:?}\". Error \"{:?}\"", opts.log_path, e),
        _ => (),
    }

    // Set-up Ctrl-C / SIGINT handler
    let running: Arc<AtomicBool> = Arc::new(AtomicBool::new(true));
    let r: Arc<AtomicBool> = running.clone();
    let m: String = opts.mode.clone();
    ctrlc::set_handler(move || {
        r.store(false, Ordering::SeqCst);
        if m == "run" {
            print!("\n");
        }
    }).expect("Error setting signal handler!");

    // Execute main logic
    match opts.mode.as_ref() {
        "run" => run(&opts, running),
        "client" => run_client(&opts, running),
        "participant" => run_participant(&opts, running),
        "check" => checker::check_last_run(opts.num_clients, opts.num_requests, opts.num_participants, &opts.log_path),
        _ => panic!("Unknown mode"),
    }
}
