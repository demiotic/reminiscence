"""Diagnostic script to check Flight server status."""

from reminiscence import Reminiscence, ReminiscenceConfig
import time

print("=" * 70)
print("DIAGNOSTIC: Checking Flight Server Status")
print("=" * 70)

# Create with default config (should have flight_enabled=True by default)
print("\n1. Creating Reminiscence with default config...")
config = ReminiscenceConfig.load()
print(f"   Config flight_enabled: {config.flight_enabled}")
print(f"   Config flight_port: {config.flight_port}")
print(f"   Config flight_host: {config.flight_host}")
print(f"   Config grpc_enabled: {config.grpc_enabled}")

cache = Reminiscence(config)

print(f"\n2. After initialization:")
print(f"   cache.flight_server is None: {cache.flight_server is None}")
print(f"   cache.flight_server value: {cache.flight_server}")
print(f"   cache.grpc_server is None: {cache.grpc_server is None}")
print(f"   cache.grpc_server value: {cache.grpc_server}")

# If gRPC auto-started, try to get capabilities
if cache.grpc_server is not None:
    print(f"\n3. gRPC server auto-started on port {config.grpc_port}")
    time.sleep(0.5)

    from reminiscence.api.client import ReminiscenceClient
    try:
        client = ReminiscenceClient(f"localhost:{config.grpc_port}")
        caps = client.get_capabilities()

        print(f"\n4. GetCapabilities() response:")
        print(f"   flight_enabled: {caps['flight_enabled']}")
        print(f"   flight_endpoint: {caps.get('flight_endpoint', 'N/A')}")

        client.close()
    except Exception as e:
        print(f"\n4. Failed to connect to gRPC: {e}")
else:
    print("\n3. gRPC server did NOT auto-start (grpc_enabled=False in config)")
    print("   Manually starting gRPC...")

    cache.start_grpc_server(port=50099, host="127.0.0.1")
    time.sleep(0.5)

    from reminiscence.api.client import ReminiscenceClient
    try:
        client = ReminiscenceClient("localhost:50099")
        caps = client.get_capabilities()

        print(f"\n4. GetCapabilities() response:")
        print(f"   flight_enabled: {caps['flight_enabled']}")
        print(f"   flight_endpoint: {caps.get('flight_endpoint', 'N/A')}")

        client.close()
    except Exception as e:
        print(f"\n4. Failed to connect to gRPC: {e}")

print("\n" + "=" * 70)
print("DIAGNOSIS COMPLETE")
print("=" * 70)

# Cleanup
if cache.grpc_server:
    cache.stop_grpc_server()
if cache.flight_server:
    cache.stop_flight_server()
