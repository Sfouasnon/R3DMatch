# Architecture

```text
R3D ingest
-> metadata extraction
-> active transform resolution
-> frame sampler
-> exposure analysis engine
-> offset decision
-> sidecar writer
-> REDLine transcode plan
-> validation output
```

The prototype keeps RED SDK integration behind a single adapter boundary so the matching logic, sidecar generation, and REDLine planning can be developed and tested before the native SDK is wired in.

