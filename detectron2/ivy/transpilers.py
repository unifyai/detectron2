import ivy
import detectron2
from detectron2 import checkpoint, config, data, engine, evaluation, export, layers, model_zoo, modeling, projects, solver, structures, tracking, utils

def to_ivy():
    return ivy.transpile(detectron2, source="torch", to="ivy")

def to_jax():
    return ivy.transpile(detectron2, source="torch", to="jax")

def to_numpy():
    return ivy.transpile(detectron2, source="torch", to="numpy")

def to_tensorflow():
    return ivy.transpile(detectron2, source="torch", to="tensorflow")

def to_torch():
    return ivy.transpile(detectron2, source="torch", to="torch")