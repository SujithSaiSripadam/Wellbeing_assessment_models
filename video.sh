batch_size=4
n_trials=2

tasks=('Task2h_A' 'Task2s_A' 'Task3_A' 'Task4_P1_A' 'Task4_P2_A' 'Task4_P3_A' 'Task5_A')
tasksv=('Task_2h' 'Task_2s' 'Task_3' 'Task_4_P1' 'Task_4_P2' 'Task_4_P3' 'Task_5')


task_index=${1}


task_name=${tasks[$task_index]}
taskv_name=${tasksv[$task_index]}

log_file="./Outputs_graph/GAT_E_${task_name}.log"

python /Users/sujithsaisripadam/Downloads/My_Internship-work/Graph/GCN/Video_train.py \
  --batch_size $batch_size \
  --n_trials $n_trials \
  --task_name $task_name &> "$log_file"