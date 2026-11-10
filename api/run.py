"""One HTTP process, one GPU job worker; do not fork CUDA workers."""
import uvicorn


def main() -> None:
    uvicorn.run("api.main:app", host="0.0.0.0", port=5522, workers=1, timeout_keep_alive=30)


if __name__ == "__main__":
    main()
