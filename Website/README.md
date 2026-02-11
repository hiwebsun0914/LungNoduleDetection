# Website

## 目录
- `backend/app.py`: FastAPI 后端，负责读取 DICOM、加载模型、推理并生成带框 GIF。
- `backend/inference.py`: 推理与可视化逻辑。
- `frontend/index.html`: 前端页面。
- `frontend/app.js`: 前端调用 `/api/predict` 并渲染结果。

## 启动
```bash
cd Website/backend
pip install -r requirements.txt
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

启动后访问:
- `http://127.0.0.1:8000/`

默认参数:
- DICOM 目录: `datasets/LIDC-IDRI-1002`
- 模型目录: `checkpoint/checkpoint-200000`
