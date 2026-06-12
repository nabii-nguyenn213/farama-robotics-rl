from __future__ import annotations
import importlib
import sys

REQUIRED_PYTHON = (3, 10)

REQUIRED_PACKAGES = {
    "gymnasium": "1.3.0",
    "numpy": "2.2.6",
    "omegaconf": "2.3.0",
    "yaml": "6.0.3",
    "torch": "2.12.0",
    "tensorboard": "2.20.0", 
    "gymnasium_robotics": "1.4.2"
}

TEST_ENVS = [
    'FetchPickAndPlaceDense-v4', 
    'FetchPushDense-v4', 
    'FetchReach-v4', 
    'FetchSlideDense-v4', 
    'HandManipulateBlock-v1', 
    'HandManipulateEgg-v1', 
    'HandManipulatePen-v1', 
    'HandReach-v3', 
    'AdroitHandDoor-v1', 
    'AdroitHandHammer-v1', 
    'AdroitHandPen-v1', 
    'AdroitHandRelocate-v1',
    'FrankaKitchen-v1'
]

def print_header(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

def normalize_version(version: str) -> str:
    return version.split("+")[0].strip()

def version_status(installed: str, expected: str) -> tuple[bool, str]:
    installed_norm = normalize_version(installed)
    expected_norm = normalize_version(expected)

    if installed == expected:
        return True, "exact"

    if installed_norm == expected_norm:
        return True, "compatible"

    return False, "mismatch"


def check_python() -> bool:
    print_header("Python Check")
    version = sys.version_info
    print(f"Detected Python: {version.major}.{version.minor}.{version.micro}")

    if (version.major, version.minor) != REQUIRED_PYTHON:
        print(
            f"✗ Expected Python {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]}.x, "
            f"but found {version.major}.{version.minor}.{version.micro}"
        )
        return False

    print("✓ Python version is compatible")
    return True


def check_packages() -> bool:
    print_header("Dependency Check")
    ok = True

    for import_name, expected_version in REQUIRED_PACKAGES.items():
        try:
            module = importlib.import_module(import_name)
            actual_version = getattr(module, "__version__", "unknown")

            matched, status = version_status(actual_version, expected_version)

            if matched and status == "exact":
                print(f"✓ {import_name:<12} {actual_version}")
            elif matched and status == "compatible":
                print(
                    f"✓ {import_name:<12} {actual_version} "
                    f"(compatible with expected {expected_version})"
                )
            else:
                print(
                    f"✗ {import_name:<12} installed={actual_version}, "
                    f"expected={expected_version}"
                )
                ok = False

        except Exception as e:
            print(f"✗ {import_name:<12} import failed -> {e}")
            ok = False

    return ok


def check_torch() -> bool:
    print_header("PyTorch Check")
    try:
        import torch

        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        x = torch.randn(3, 3)
        y = x @ x
        print("✓ Torch tensor op OK")
        return True
    except Exception as e:
        print(f"✗ PyTorch check failed -> {e}")
        return False




def check_gym_envs() -> bool:
    print_header("Gymnasium Environment Check")
    try:
        import gymnasium as gym
    except Exception as e:
        print(f"✗ Could not import gymnasium -> {e}")
        return False

    all_ok = True

    for env_name in TEST_ENVS:
        try:
            if env_name == "LunarLander-v3" : 
                env=gym.make(env_name, continuous=True)
            else: 
                env = gym.make(env_name)
            obs, info = env.reset()
            action = env.action_space.sample()
            step_result = env.step(action)
            env.close()

            if len(step_result) != 5:
                raise RuntimeError(
                    f"Unexpected step return length for {env_name}: {len(step_result)}"
                )

            print(f"✓ {env_name:<24} reset/step OK")
        except Exception as e:
            print(f"✗ {env_name:<24} failed -> {e}")
            all_ok = False

    return all_ok


def main() -> None:
    print("Running project setup test...")

    results = {
        "python": check_python(),
        "packages": check_packages(),
        "torch": check_torch(),
        "gym_envs": check_gym_envs(),
    }

    print_header("Summary")
    for name, passed in results.items():
        print(f"{'✓' if passed else '✗'} {name}")

    if all(results.values()):
        print("\nAll checks passed. Environment is set up correctly.")
    else:
        print("\nSome checks failed. Please review the messages above.")


if __name__ == "__main__":
    main()
