"""Start a Reminiscence server with both gRPC and Flight enabled for CLI testing."""

from reminiscence import Reminiscence, ReminiscenceConfig
import time

print("Starting Reminiscence server with gRPC and Flight...")
print("=" * 70)

# Configure with both servers enabled
config = ReminiscenceConfig.load()
config.grpc_enabled = True
config.flight_enabled = True
config.grpc_port = 50051
config.flight_port = 8081

cache = Reminiscence(config)

print(f"Flight server: {cache.flight_server is not None}")
print(f"gRPC server: {cache.grpc_server is not None}")
print()
print("Servers running:")
print(f"  gRPC:  http://localhost:50051")
print(f"  Flight: grpc://127.0.0.1:8081")
print()
print("Press Ctrl+C to stop...")

try:
    # Keep running
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nStopping servers...")
    if cache.grpc_server:
        cache.stop_grpc_server()
    if cache.flight_server:
        cache.stop_flight_server()
    print("Servers stopped.")
