import sys, argparse
name, delay, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
sys.argv = sys.argv[:1]
import controller_study as cs
cs.tune(argparse.Namespace(controllers=[name], rounds=3, K=12, M=6, delay=delay, out=out))
