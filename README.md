# Machine learning projects

- [Number classification](number%20classification/README.md): my handwritten digit classifier using MNIST, PyTorch, and Gradio.
- `fashion mnist/`: space for my next project.

## Run the number classifier locally

From this repository folder in PowerShell:

```powershell
.\gradio-env\Scripts\Activate.ps1
cd "number classification"
python app.py
```

The existing virtual environment stays at the repository root. Run training from inside `number classification` too, so its relative data and model paths resolve correctly.

## Render deployment

Open the existing number-classifier web service in Render, then go to **Settings > Build & Deploy**:

| Setting | Value |
| --- | --- |
| Branch | `main` |
| Root Directory | `number classification` |
| Build Command | `pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt` |
| Start Command | `python app.py` |

Enter the root directory without quotation marks. Commands run from inside it. The build command installs CPU-only PyTorch first because this deployment does not need CUDA.

Save the settings, verify the commands in Render's update dialog, then use **Manual Deploy > Deploy latest commit**. If an old dependency installation is still causing errors, use **Clear build cache & deploy**.

The app already binds to `0.0.0.0` and reads Render's `PORT` environment variable. Keep `MNIST_MLP.pth` in the classifier folder; deployment loads these saved weights and does not run training.

If deployment still fails, inspect the first error in the build or runtime logs. These settings fix paths after the move and install the app dependencies; they do not establish the cause of any other existing deployment failure.

Render documentation: https://render.com/docs/monorepo-support
