from . import * 

if __name__ == '__main__':
    args = olpy_parse_args()

    run_experiments(
        args.train_set, args.test_set, args.models, args.n, args.label,
        args.bias, args.use_weights, args.weights, args.cv, 
        args.dump_dir, args.v, args.o, args.s,
    )
