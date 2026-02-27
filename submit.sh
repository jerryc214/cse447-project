#!/usr/bin/env bash
set -x
set -e

WORK_DIR=${SUBMIT_WORK_DIR:-work}

rm -rf submit submit.zip
mkdir -p submit

# submit team.txt
if [ -f team.txt ]; then
  cp team.txt submit/team.txt
else
  printf "Name1,NetID1\nName2,NetID2\nName3,NetID3\n" > submit/team.txt
  echo "WARNING: team.txt not found in repo root; wrote placeholder submit/team.txt"
fi

# make predictions on example data and submit in pred.txt
if [ ! -f "$WORK_DIR/model.checkpoint" ]; then
  echo "ERROR: $WORK_DIR/model.checkpoint not found. Set SUBMIT_WORK_DIR or prepare checkpoint."
  exit 1
fi
python src/myprogram.py test --work_dir "$WORK_DIR" --test_data example/input.txt --test_output submit/pred.txt

# submit docker file
cp Dockerfile submit/Dockerfile

# submit source code
cp -r src submit/src
find submit/src -type d -name "__pycache__" -prune -exec rm -rf {} +
find submit/src -type f -name "*.pyc" -delete

# submit checkpoints
cp -r "$WORK_DIR" submit/work

# make zip file
zip -r submit.zip submit
