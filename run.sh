#!/bin/bash
#!/user/bin/env python3

python3 src/data/get_data.py $1 $2
python3 src/data/get_citations.py $1 $2
python3 src/models/log_reg.py $1 $2
python3 src/models/decision_tree.py $1 $2
python3 src/models/random_forest.py $1 $2