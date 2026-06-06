# LitePT

## 环境准备

```bash
conda activate litept
export PYTHONPATH=./
```

## CUDA Extensions (libs/)

编译前修改 `libs/pointrope/setup.py` 中 `all_cuda_archs` 匹配 GPU：

| GPU | compute capability |
|---|---|
| RTX 3090 | sm_86 |
| RTX 4080 | sm_89 |
| H100 | sm_90 |

```bash
cd libs/pointrope && python setup.py install && cd ../..
cd libs/pointops && python setup.py install && cd ../..
```

## Agent skills

### Issue tracker

Issues tracked in GitHub (`zhou-zgg/LitePT_forest`). See `docs/agents/issue-tracker.md`.

### Triage labels

Default label vocabulary (needs-triage, needs-info, ready-for-agent, ready-for-human, wontfix). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout. See `docs/agents/domain.md`.