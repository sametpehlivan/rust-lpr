"""Tiny local test receiver for NetLpr Rust's newline-delimited TCP events.
Run: python examples/tcp_receiver.py
Do not expose to the internet: no authentication/TLS in this demo.
"""
import json
import socket

HOST, PORT = "127.0.0.1", 9000
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen()
    print(f"Listening for NetLpr events on {HOST}:{PORT}")
    while True:
        connection, address = server.accept()
        with connection, connection.makefile("rb") as stream:
            print("Connected:", address)
            for raw_line in stream:
                try:
                    event = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    print("Invalid JSON:", exc)
                    continue
                print(event.get("source_id"), event.get("tracked_id"), event.get("plate"))
