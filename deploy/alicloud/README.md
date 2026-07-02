# RankAI — Qwen Cloud Hackathon Submission (Track 4: Autopilot Agent)

Alibaba Cloud deployment for RankAI — an end-to-end agentic hiring pipeline
that automatically parses resumes, enriches career trajectories, runs a
3-persona multi-agent panel on Alibaba Cloud's Qwen Cloud LLMs, audits the
scores for demographic bias, and persists a ranked shortlist with a human
in the loop.

This directory contains:

- `docker-compose.yml` — single-server Compose stack (FastAPI + ChromaDB + Nginx)
- `Dockerfile` — production image for the FastAPI backend
- `nginx.conf` — reverse proxy with a `/healthz`, `/readyz` and `/metrics` surface
- `ecs-terraform.tf` — Terraform skeleton for Alibaba Cloud **ECS** (IaaS)
- `fc-template.yml` — Alibaba Cloud **Function Compute** HTTP trigger template
- `alicloud-oss-proof.py` — **Proof-of-deployment** script that lists OSS
  buckets via the official `alibabacloud_oss_v2` SDK. Used during the demo to
  demonstrate live Alibaba Cloud integration.

## Quick Start (Docker Compose)

```bash
cp .env.example .env
# Edit .env: DASHSCOPE_API_KEY=sk-..., LLM_BACKEND=qwen_cloud, etc.
docker compose -f deploy/alicloud/docker-compose.yml --env-file .env up -d
curl http://localhost:8000/healthz   # → {"status":"ok","backend":"qwen_cloud"}
curl -X POST http://localhost:8000/api/run -H 'Content-Type: application/json' \
     -d '{"job_description":"data/sample_job_description.json","candidates_dir":"data/candidates"}'
```

## Deployment Targets

| Target | File | Use When |
|--------|------|----------|
| ECS (VM) | `ecs-terraform.tf` | You want full control, custom models, persistent ChromaDB volume |
| Function Compute | `fc-template.yml` | Serverless, pay-per-request, cold-start tolerant |
| Container Service (ACK) | `docker-compose.yml` + `Dockerfile` | K8s-style orchestration on Alibaba |

For the hackathon we recommend **ECS + docker-compose** for the live demo
because it gives a stable public IP and persistent ChromaDB storage. The
`fc-template.yml` is included as a stretch deliverable for production-grade
elastic scaling.

## Proof of Alibaba Cloud Deployment

```bash
python scripts/alicloud_deployment_proof.py
```

Hits the live Alibaba Cloud OSS endpoint and lists the buckets owned by the
demo deployment. The transcript is captured as `output/alicloud_proof.json`.

See `docs/HACKATHON_DEPLOYMENT_PROOF.md` for the recorded run.
