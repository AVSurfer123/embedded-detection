# Networking Stack for Centralized Learning

Here is our network architecture diagram for how we are performing centralized learning with edge devices.

![](architecture.png "Network Architecture Diagram")

We use Rust for the network communication between edge client and central server, sending data through TCP sockets.
We use the disk to perform IPC between processes on the client and server.
These processes will use a Python3 wrapper to read/write data to the disk which is then picked up by the networking code and sent/received.

## Installation
First install Rust with
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```
Then run 
```
cargo run --bin server
```
to start the server-side networking node and
```
cargo run --bin client
```
to start the client-side networking node.

Finally, interface with the networking node using Python functions inside of `data_utils.py`:
```python
Client methods:
write_image(image: np.ndarray, label: Any)
load_model(last_time: float) -> Tuple[Optional[tf.lite.Interpreter], float]

Server methods:
read_new_images(last_time: float) -> Tuple[List[Tuple[np.ndarray, Any]], float]
save_model(tflite_model)
```
The documentation for these functions are in their docstrings.
