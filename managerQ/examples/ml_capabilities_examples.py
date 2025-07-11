"""
ML Capabilities Examples

This file demonstrates how to use the advanced ML capabilities in the Q2 platform.
Examples include API usage and agent interactions.
"""

import asyncio
import requests
import json
import base64

# ===== API EXAMPLES =====

class MLCapabilitiesExamples:
    """Examples for using ML capabilities via API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.ml_base_url = f"{base_url}/v1/ml"
    
    # ===== FEDERATED LEARNING EXAMPLES =====
    
    def start_federated_learning_example(self):
        """Example: Start a federated learning session"""
        
        # Example configuration for image classification
        request_data = {
            "model_architecture": "CNN_ImageClassifier",
            "dataset_config": {
                "dataset_type": "image_classification",
                "num_classes": 10,
                "image_size": [224, 224],
                "data_source": "distributed_agents"
            },
            "training_config": {
                "epochs": 10,
                "batch_size": 32,
                "learning_rate": 0.001,
                "optimizer": "adam"
            },
            "aggregation_strategy": "federated_averaging",
            "privacy_config": {
                "differential_privacy": True,
                "epsilon": 1.0,
                "delta": 1e-5
            }
        }
        
        response = requests.post(
            f"{self.ml_base_url}/federated-learning/start",
            json=request_data
        )
        
        if response.status_code == 200:
            result = response.json()
            session_id = result["session_id"]
            print(f"✅ Federated learning started: {session_id}")
            return session_id
        else:
            print(f"❌ Error: {response.text}")
            return None
    
    def check_federated_learning_status(self, session_id: str):
        """Example: Check federated learning status"""
        
        response = requests.get(
            f"{self.ml_base_url}/federated-learning/status/{session_id}"
        )
        
        if response.status_code == 200:
            status = response.json()
            print(f"📊 Federated Learning Status:")
            print(f"   Status: {status.get('status', 'unknown')}")
            print(f"   Rounds: {len(status.get('rounds', []))}")
            print(f"   Model Version: {status.get('current_model_version', 'N/A')}")
            return status
        else:
            print(f"❌ Error checking status: {response.text}")
            return None
    
    # ===== AUTOML EXAMPLES =====
    
    def start_automl_experiment_example(self):
        """Example: Start an AutoML experiment"""
        
        request_data = {
            "experiment_name": "Customer_Churn_Prediction",
            "model_type": "classification",
            "dataset_config": {
                "dataset_path": "/data/customer_data.csv",
                "target_column": "churn",
                "feature_columns": ["age", "tenure", "monthly_charges", "total_charges"],
                "categorical_columns": ["gender", "partner", "dependents"]
            },
            "optimization_objective": "f1",
            "n_trials": 50,
            "timeout_hours": 2
        }
        
        response = requests.post(
            f"{self.ml_base_url}/automl/start",
            json=request_data
        )
        
        if response.status_code == 200:
            result = response.json()
            experiment_id = result["experiment_id"]
            print(f"🔬 AutoML experiment started: {experiment_id}")
            return experiment_id
        else:
            print(f"❌ Error: {response.text}")
            return None
    
    def check_automl_status(self, experiment_id: str):
        """Example: Check AutoML experiment status"""
        
        response = requests.get(
            f"{self.ml_base_url}/automl/status/{experiment_id}"
        )
        
        if response.status_code == 200:
            status = response.json()
            print(f"🧪 AutoML Experiment Status:")
            print(f"   Status: {status.get('status', 'unknown')}")
            print(f"   Progress: {status.get('trials_completed', 0)}/{status.get('total_trials', 0)}")
            print(f"   Best Score: {status.get('best_score', 'N/A')}")
            return status
        else:
            print(f"❌ Error checking status: {response.text}")
            return None
    
    def get_automl_results(self, experiment_id: str):
        """Example: Get AutoML experiment results"""
        
        response = requests.get(
            f"{self.ml_base_url}/automl/results/{experiment_id}"
        )
        
        if response.status_code == 200:
            results = response.json()
            print(f"📈 AutoML Results:")
            print(f"   Total Models: {len(results.get('results', []))}")
            
            # Show top 3 models
            top_models = results.get('results', [])[:3]
            for i, model in enumerate(top_models, 1):
                print(f"   #{i} {model.get('model_name', 'Unknown')}: {model.get('performance_metrics', {}).get('test_score', 'N/A')}")
            
            return results
        else:
            print(f"❌ Error getting results: {response.text}")
            return None
    
    # ===== REINFORCEMENT LEARNING EXAMPLES =====
    
    def start_rl_training_example(self):
        """Example: Start RL training for workflow optimization"""
        
        request_data = {
            "agent_name": "WorkflowOptimizer_v1",
            "environment_type": "workflow_optimization",
            "algorithm": "ppo",
            "training_config": {
                "learning_rate": 3e-4,
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10
            },
            "environment_config": {
                "max_steps": 100,
                "reward_shaping": True,
                "state_normalization": True
            },
            "total_timesteps": 100000
        }
        
        response = requests.post(
            f"{self.ml_base_url}/rl/start-training",
            json=request_data
        )
        
        if response.status_code == 200:
            result = response.json()
            session_id = result["session_id"]
            print(f"🎮 RL training started: {session_id}")
            return session_id
        else:
            print(f"❌ Error: {response.text}")
            return None
    
    def check_rl_training_status(self, session_id: str):
        """Example: Check RL training status"""
        
        response = requests.get(
            f"{self.ml_base_url}/rl/training-status/{session_id}"
        )
        
        if response.status_code == 200:
            status = response.json()
            print(f"🤖 RL Training Status:")
            print(f"   Status: {status.get('status', 'unknown')}")
            print(f"   Progress: {status.get('progress', 0):.1%}")
            print(f"   Current Reward: {status.get('current_reward', 'N/A')}")
            print(f"   Best Reward: {status.get('best_reward', 'N/A')}")
            return status
        else:
            print(f"❌ Error checking status: {response.text}")
            return None
    
    # ===== MULTIMODAL AI EXAMPLES =====
    
    def classify_image_example(self, image_path: str):
        """Example: Classify an image"""
        
        try:
            # Read and encode image
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            # Upload and classify
            files = {'file': ('image.jpg', image_data, 'image/jpeg')}
            form_data = {
                'modality': 'image',
                'content_type': 'image/jpeg',
                'agent_id': 'default'
            }
            
            response = requests.post(
                f"{self.ml_base_url}/multimodal/classify-image",
                files=files,
                data=form_data
            )
            
            if response.status_code == 200:
                result = response.json()
                request_id = result["request_id"]
                print(f"🖼️ Image classification started: {request_id}")
                return request_id
            else:
                print(f"❌ Error: {response.text}")
                return None
                
        except FileNotFoundError:
            print(f"❌ Image file not found: {image_path}")
            return None
    
    def transcribe_audio_example(self, audio_path: str):
        """Example: Transcribe audio"""
        
        try:
            # Read and encode audio
            with open(audio_path, 'rb') as f:
                audio_data = f.read()
            
            # Upload and transcribe
            files = {'audio': ('audio.wav', audio_data, 'audio/wav')}
            form_data = {'agent_id': 'default'}
            
            response = requests.post(
                f"{self.ml_base_url}/multimodal/transcribe-audio",
                files=files,
                data=form_data
            )
            
            if response.status_code == 200:
                result = response.json()
                request_id = result["request_id"]
                print(f"🎵 Audio transcription started: {request_id}")
                return request_id
            else:
                print(f"❌ Error: {response.text}")
                return None
                
        except FileNotFoundError:
            print(f"❌ Audio file not found: {audio_path}")
            return None
    
    def analyze_sentiment_example(self, text: str):
        """Example: Analyze sentiment"""
        
        form_data = {
            'text': text,
            'agent_id': 'default'
        }
        
        response = requests.post(
            f"{self.ml_base_url}/multimodal/analyze-sentiment",
            data=form_data
        )
        
        if response.status_code == 200:
            result = response.json()
            request_id = result["request_id"]
            print(f"💭 Sentiment analysis started: {request_id}")
            return request_id
        else:
            print(f"❌ Error: {response.text}")
            return None
    
    def check_multimodal_result(self, request_id: str):
        """Example: Check multimodal processing result"""
        
        response = requests.get(
            f"{self.ml_base_url}/multimodal/result/{request_id}"
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"🎯 Multimodal Result:")
            print(f"   Task: {result.get('result', {}).get('task', 'unknown')}")
            
            # Handle different result types
            result_data = result.get('result', {})
            if 'predicted_class' in result_data:
                print(f"   Predicted Class: {result_data['predicted_class']}")
                print(f"   Confidence: {result_data.get('confidence', 'N/A')}")
            elif 'transcription' in result_data:
                print(f"   Transcription: {result_data['transcription']}")
            elif 'sentiment' in result_data:
                print(f"   Sentiment: {result_data['sentiment']}")
                print(f"   Confidence: {result_data.get('confidence', 'N/A')}")
            
            return result
        else:
            print(f"❌ Error getting result: {response.text}")
            return None
    
    # ===== COMPREHENSIVE EXAMPLES =====
    
    def run_complete_automl_workflow(self):
        """Example: Complete AutoML workflow from start to finish"""
        
        print("🚀 Starting Complete AutoML Workflow")
        
        # Step 1: Start experiment
        experiment_id = self.start_automl_experiment_example()
        if not experiment_id:
            return
        
        # Step 2: Monitor progress
        import time
        while True:
            status = self.check_automl_status(experiment_id)
            if not status:
                break
            
            if status.get('status') == 'completed':
                print("✅ Experiment completed!")
                break
            elif status.get('status') == 'failed':
                print("❌ Experiment failed!")
                break
            
            print("⏳ Waiting for completion...")
            time.sleep(30)  # Check every 30 seconds
        
        # Step 3: Get results
        results = self.get_automl_results(experiment_id)
        if results:
            print("🎉 AutoML workflow completed successfully!")
    
    def run_multimodal_pipeline(self, text: str):
        """Example: Process text through multimodal pipeline"""
        
        print("🌟 Starting Multimodal Pipeline")
        
        # Analyze sentiment
        sentiment_id = self.analyze_sentiment_example(text)
        if sentiment_id:
            # Wait a bit and get result
            import time
            time.sleep(5)
            sentiment_result = self.check_multimodal_result(sentiment_id)
            print(f"✨ Sentiment analysis completed!")
        
        print("🎊 Multimodal pipeline completed!")

# ===== AGENT INTERACTION EXAMPLES =====

class AgentMLExamples:
    """Examples for using ML capabilities through agent interactions"""
    
    def __init__(self, manager_url: str = "http://localhost:8000"):
        self.manager_url = manager_url
    
    def send_task_to_ml_agent(self, prompt: str):
        """Send a task to the ML specialist agent"""
        
        task_data = {
            "prompt": prompt,
            "agent_type": "ml_specialist",
            "priority": "normal"
        }
        
        response = requests.post(
            f"{self.manager_url}/v1/tasks",
            json=task_data
        )
        
        if response.status_code == 200:
            result = response.json()
            task_id = result["task_id"]
            print(f"📬 Task sent to ML agent: {task_id}")
            return task_id
        else:
            print(f"❌ Error sending task: {response.text}")
            return None
    
    def check_task_result(self, task_id: str):
        """Check the result of a task"""
        
        response = requests.get(
            f"{self.manager_url}/v1/tasks/{task_id}"
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"📋 Task Result:")
            print(f"   Status: {result.get('status', 'unknown')}")
            if result.get('status') == 'completed':
                print(f"   Result: {result.get('result', 'N/A')}")
            return result
        else:
            print(f"❌ Error checking task: {response.text}")
            return None

# ===== EXAMPLE PROMPTS FOR ML AGENT =====

ML_AGENT_EXAMPLE_PROMPTS = {
    "automl_simple": """
    I have a CSV dataset with customer information and I want to predict customer churn. 
    Can you help me train a model using AutoML? The dataset has columns: age, tenure, 
    monthly_charges, total_charges, gender, partner, dependents, and churn (target).
    """,
    
    "federated_learning": """
    I want to train an image classification model across multiple agents without 
    sharing raw data. Can you set up federated learning with privacy preservation?
    The model should classify images into 10 categories.
    """,
    
    "workflow_optimization": """
    Our workflow for processing customer orders is taking too long. Can you use 
    reinforcement learning to optimize it? The workflow has steps for validation, 
    inventory check, payment processing, and fulfillment.
    """,
    
    "sentiment_analysis": """
    Please analyze the sentiment of this text: "I absolutely love the new features 
    in this product! The user interface is intuitive and the performance is amazing."
    """,
    
    "ml_capabilities_overview": """
    What ML capabilities are available in the platform? Can you give me a summary 
    of all the machine learning services and their current status?
    """
}

# ===== MAIN DEMO FUNCTION =====

def run_ml_demo():
    """Run a comprehensive demo of ML capabilities"""
    
    print("🎯 Q2 Platform ML Capabilities Demo")
    print("=" * 50)
    
    # Initialize examples
    api_examples = MLCapabilitiesExamples()
    agent_examples = AgentMLExamples()
    
    print("\n1️⃣ Testing API Endpoints...")
    
    # Test sentiment analysis
    sentiment_id = api_examples.analyze_sentiment_example(
        "The Q2 platform is revolutionary! I love the ML capabilities."
    )
    
    if sentiment_id:
        import time
        time.sleep(3)  # Wait for processing
        api_examples.check_multimodal_result(sentiment_id)
    
    print("\n2️⃣ Testing Agent Interactions...")
    
    # Test ML agent
    task_id = agent_examples.send_task_to_ml_agent(
        ML_AGENT_EXAMPLE_PROMPTS["ml_capabilities_overview"]
    )
    
    if task_id:
        import time
        time.sleep(5)  # Wait for processing
        agent_examples.check_task_result(task_id)
    
    print("\n🎉 Demo completed!")
    print("\nTry these example prompts with the ML agent:")
    for name, prompt in ML_AGENT_EXAMPLE_PROMPTS.items():
        print(f"\n🔹 {name.replace('_', ' ').title()}:")
        print(f"   \"{prompt[:100]}...\"")

if __name__ == "__main__":
    # Run the demo
    run_ml_demo()
    
    # Print usage instructions
    print("\n" + "=" * 60)
    print("📚 USAGE INSTRUCTIONS")
    print("=" * 60)
    print("""
To use the ML capabilities:

1. 🌐 Via API:
   - Use the MLCapabilitiesExamples class
   - Send requests to /v1/ml/* endpoints
   - Monitor progress and get results

2. 🤖 Via ML Agent:
   - Send natural language prompts to the ML specialist agent
   - Use the AgentMLExamples class
   - The agent will handle tool selection and execution

3. 🔧 Integration:
   - Import ml_tools in your agents
   - Use convenience tools like train_model_on_data
   - Check ML capabilities summary

4. 📈 Best Practices:
   - Start with simple tasks first
   - Monitor long-running processes
   - Use appropriate data formats (JSON, base64)
   - Check service status before starting tasks
    """) 