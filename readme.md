# Setting up a Proof of Concept (PoC) MLOps Platform on MacOS

This guide helps you set up a PoC MLOps platform on Mac with a local Kubernetes cluster using Seldon Core.

## Prerequisites

- **MacOS**: Ensure your Mac is up-to-date.
- **Terminal**: Familiarity with terminal commands.

## Installation

1. **Install Docker Desktop**:
    - [Download Docker Desktop for Mac](https://docs.docker.com/desktop/mac/install/).
    - Enable Kubernetes in Docker Desktop preferences.

2. **Install kubectl**:
   ```bash
   brew install kubectl
   ```

3. **Install Helm**:
   ```bash
   brew install helm
   ```

4. **Verify Kubernetes**:
   ```bash
   kubectl cluster-info
   ```

5. **Install Seldon Core**:
   ```bash
   helm repo add seldon https://storage.googleapis.com/seldon-charts
   helm repo update
   kubectl create namespace seldon-system
   helm install seldon-core seldon/seldon-core-operator --namespace seldon-system
   ```

6. **Install ZenML (Optional)**:
   ```bash
   pip install zenml
   zenml init
   ```

7. **Prepare ML Model**:
    - Select a dataset (e.g., [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)).
    - Develop a model using libraries like pandas, scikit-learn, etc.

8. **Containerize and Deploy Model**:
    - Write a Dockerfile.
    - Build Docker image:
      ```bash
      docker build -t your-model-image-name .
      ```
    - Deploy using Seldon Core and a YAML file.

9. **Test Deployment**:
    - Send requests to the model using `curl` or Postman.

10. **Monitor and Manage (Optional)**:
    - Install and access the Kubernetes dashboard for visualization.

## Notes

- Monitor resource usage on your Mac.
- Experiment with different model architectures and datasets for PoC.

```