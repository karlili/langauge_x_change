language-x-change Transcribe Restful API

This subproject is aimed to provide a way to access the transcription model via a RESRful API

# Prerequisite 

- You will need to have Docker running on your machine. It is available for MacOS and Window machine. You can download a copy from here (https://www.docker.com/products/docker-desktop/). Follow the instruction to install, and restart your machine if necessary

- Verify if the installation is good by opening up the Terminal (from MacOS), then type `docker`, you should see the following coming up if the installation is good

```
(langauge_x_change) kenny@bunnies-family-Mac-Studio langauge_x_change % docker

Usage:  docker [OPTIONS] COMMAND

A self-sufficient runtime for containers

Common Commands:
  run         Create and run a new container from an image
  exec        Execute a command in a running container
  ps          List containers
  build       Build an image from a Dockerfile
  pull        Download an image from a registry
  push        Upload an image to a registry
  images      List images
  login       Log in to a registry
  logout      Log out from a registry
  search      Search Docker Hub for images
  version     Show the Docker version information
  info        Display system-wide information

Management Commands:
  builder     Manage builds
  buildx*     Docker Buildx (Docker Inc., v0.12.0-desktop.2)
  compose*    Docker Compose (Docker Inc., v2.23.3-desktop.2)
  container   Manage containers
  context     Manage contexts
  dev*        Docker Dev Environments (Docker Inc., v0.1.0)
  extension*  Manages Docker extensions (Docker Inc., v0.2.21)
  feedback*   Provide feedback, right in your terminal! (Docker Inc., 0.1)
  image       Manage images
  init*       Creates Docker-related starter files for your project (Docker Inc., v0.1.0-beta.10)
  manifest    Manage Docker image manifests and manifest lists
  network     Manage networks
  plugin      Manage plugins
  sbom*       View the packaged-based Software Bill Of Materials (SBOM) for an image (Anchore Inc., 0.6.0)
  scan*       Docker Scan (Docker Inc., v0.26.0)
  scout*      Docker Scout (Docker Inc., v1.2.0)
  system      Manage Docker
  trust       Manage trust on Docker images
  volume      Manage volumes

Swarm Commands:
  swarm       Manage Swarm

Commands:
  attach      Attach local standard input, output, and error streams to a running container
  commit      Create a new image from a container's changes
  cp          Copy files/folders between a container and the local filesystem
  create      Create a new container
  diff        Inspect changes to files or directories on a container's filesystem
  events      Get real time events from the server
  export      Export a container's filesystem as a tar archive
  history     Show the history of an image
  import      Import the contents from a tarball to create a filesystem image
  inspect     Return low-level information on Docker objects
  kill        Kill one or more running containers
  load        Load an image from a tar archive or STDIN
  logs        Fetch the logs of a container
  pause       Pause all processes within one or more containers
  port        List port mappings or a specific mapping for the container
  rename      Rename a container
  restart     Restart one or more containers
  rm          Remove one or more containers
  rmi         Remove one or more images
  save        Save one or more images to a tar archive (streamed to STDOUT by default)
  start       Start one or more stopped containers
  stats       Display a live stream of container(s) resource usage statistics
  stop        Stop one or more running containers
  tag         Create a tag TARGET_IMAGE that refers to SOURCE_IMAGE
  top         Display the running processes of a container
  unpause     Unpause all processes within one or more containers
  update      Update configuration of one or more containers
  wait        Block until one or more containers stop, then print their exit codes

Global Options:
      --config string      Location of client config files (default "/Users/kenny/.docker")
  -c, --context string     Name of the context to use to connect to the daemon (overrides DOCKER_HOST env var and default context set with "docker context use")
  -D, --debug              Enable debug mode
  -H, --host list          Daemon socket to connect to
  -l, --log-level string   Set the logging level ("debug", "info", "warn", "error", "fatal") (default "info")
      --tls                Use TLS; implied by --tlsverify
      --tlscacert string   Trust certs signed only by this CA (default "/Users/kenny/.docker/ca.pem")
      --tlscert string     Path to TLS certificate file (default "/Users/kenny/.docker/cert.pem")
      --tlskey string      Path to TLS key file (default "/Users/kenny/.docker/key.pem")
      --tlsverify          Use TLS and verify the remote
  -v, --version            Print version information and quit

Run 'docker COMMAND --help' for more information on a command.

For more help on how to use Docker, head to https://docs.docker.com/go/guides/
(langauge_x_change) kenny@bunnies-family-Mac-Studio langauge_x_change % 
```

# Start up

Assuming your current folder is `dockerise_api`, run this command `docker compose up`.

The whole setup process should be automated and do not require any intervention.

You would see something like this when the process is running fine.

```
vscode ➜ /workspaces/langauge_x_change/dockerise_api (develop) $ docker compose up
[+] Building 6.6s (4/13)                                                              docker:default
 => [api internal] load build definition from Dockerfile                                        0.3s
 ...
 ...
 ...
 
[+] Running 2/2
 ✔ Network dockerise_api_default  Created                                                       0.3s 
 ✔ Container dockerise_api-api-1  Created                                                       1.1s 
Attaching to api-1
api-1  | INFO:     Will watch for changes in these directories: ['/app']
api-1  | INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
api-1  | INFO:     Started reloader process [7] using WatchFiles
api-1  | INFO:     Started server process [9]
api-1  | INFO:     Waiting for application startup.
api-1  | INFO:     Application startup complete. 
 
 
```

When you see the last message shown 'Application startup complete.' Then you can head to the browser and navigate to http://127.0.0.1:8000/docs and you can see all the available endpoints. For the time being, we have `/uploads`


# Multiple Instances

If your machine has enough resources, you can run multiple instances of the API with docker compose

Assuming your current folder is `dockerise_api`, run this command `docker compose up --scale language-x-change-api=<NUM_OF_INSTANCE>`.

If the `NUM_OF_INSTANCE` is 3, it will run 3 instances of the API. you can double check with the command `docker ps -a`, and it will give you something like this.
```
a5c2b2fe4769   dockerise_api-language-x-change-api    "/bin/sh -c 'pip ins…"   4 minutes ago   Up 4 minutes                   0.0.0.0:60235->8000/tcp          dockerise_api-language-x-change-api-3
eb2adf15b916   dockerise_api-language-x-change-api    "/bin/sh -c 'pip ins…"   4 minutes ago   Up 4 minutes                   0.0.0.0:60236->8000/tcp          dockerise_api-language-x-change-api-2
a41c49fccaf9   dockerise_api-language-x-change-api    "/bin/sh -c 'pip ins…"   9 minutes ago   Up 4 minutes                   0.0.0.0:60238->8000/tcp          dockerise_api-language-x-change-api-1

```