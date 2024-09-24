# federated-neuromophic
This is a project that will explore the use of neurotrophic learning for privacy aware federated learning


## Everyone:
Change the path in globals.py where you have MNIST downloaded

## Project Structure

``` MD
federated-neuromorphic/
├── 0_examples/ 
│   └── ... demo examples 
├── data/ 
│   └── mnist_loader.py 
├── models/
│   ├── __init__.py
│   ├── simple_model.py
│   ├── neuromorphic/
│   │   ├── __init__.py
│   │   ├── feedback_alignment_model.py
│   │   └── other_technique_model.py
│   └── federated/
│       ├── __init__.py
│       ├── federated_model.py
│       ├── federated_neuromorphic_model.py
│       ├── client.py
│       └── server.py
├── training/
│   ├── __init__.py
│   ├── simple_training.py
│   ├── neuromorphic_training.py
│   └── federated_training/
│       ├── __init__.py
│       ├── federated_training.py
│       ├── federated_neuromorphic_training.py
│       ├── client_update.py
│       └── server_aggregation.py
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py
│   └── analysis.py
├── utils/
│   ├── __init__.py
│   └── helper_functions.py
├── main.py
├── requirements.txt
└── README.md
```