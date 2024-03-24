def print_args(args):
    print("\033[1m" + "Basic Config" + "\033[0m")
    print(f'  {"Is Training:":<20}{args.is_training:<20}')
    print()

    print("\033[1m" + "Data Loader" + "\033[0m")
    print(f'  {"Root Path:":<20}{args.root_path:<20}')
    print(f'  {"Data Path:":<20}{args.data_path:<20}')
    print(f'  {"Checkpoints:":<20}{args.checkpoints:<20}')
    print()

    print("\033[1m" + "Model Parameters" + "\033[0m")
    print(f'  {"Top k:":<20}{args.top_k:<20}{"Num Kernels:":<20}{args.num_kernels:<20}')
    print(f'  {"Enc In:":<20}{args.enc_in:<20}{"C Out:":<20}{args.c_out:<20}')
    print(f'  {"d model:":<20}{args.d_model:<20}{"e layers:":<20}{args.e_layers:<20}')
    print(f'  {"Dropout:":<20}{args.dropout:<20}{"Activation:":<20}{args.activation:<20}')
    print()

    print("\033[1m" + "Run Parameters" + "\033[0m")
    print(f'  {"Num Workers:":<20}{args.num_workers:<20}{"Itr:":<20}{args.itr:<20}')
    print(f'  {"Train Epochs:":<20}{args.train_epochs:<20}{"Batch Size:":<20}{args.batch_size:<20}')
    print(f'  {"Patience:":<20}{args.patience:<20}{"Learning Rate:":<20}{args.learning_rate:<20}')
    print(f'  {"Loss:":<20}{args.loss:<20}')
    print()

    print("\033[1m" + "GPU" + "\033[0m")
    print(f'  {"Use GPU:":<20}{args.use_gpu:<20}{"GPU:":<20}{args.gpu:<20}')
    print(f'  {"Devices:":<20}{args.devices:<20}')
    print()
