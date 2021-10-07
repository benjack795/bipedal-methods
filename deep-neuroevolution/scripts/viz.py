import click


@click.command()
@click.argument('env_id')
@click.argument('policy_file')
@click.option('--record', is_flag=True)
@click.option('--stochastic', is_flag=True)
@click.option('--extra_kwargs')
def main(env_id, policy_file, record, stochastic, extra_kwargs):
    import gym
    from gym import wrappers
    import tensorflow as tf
    from es_distributed.policies import MujocoPolicy, ESAtariPolicy
    from es_distributed.atari_wrappers import ScaledFloatFrame, wrap_deepmind
    from es_distributed.es import get_ref_batch
    import numpy as np

    is_atari_policy = "NoFrameskip" in env_id

    env = gym.make(env_id)
    if is_atari_policy:
        env = wrap_deepmind(env)

    if record:
        import uuid
        env = wrappers.Monitor(env, '/tmp/' + str(uuid.uuid4()), force=True)

    if extra_kwargs:
        import json
        extra_kwargs = json.loads(extra_kwargs)

    with tf.Session():
        if is_atari_policy:
            pi = ESAtariPolicy.Load(policy_file, extra_kwargs=extra_kwargs)
            pi.set_ref_batch(get_ref_batch(env, batch_size=128))
        else:
            pi = MujocoPolicy.Load(policy_file, extra_kwargs=extra_kwargs)
            
        #BJ - modified to visualise and create output csv

######################################
        data_array = []
        # visualise with noise (0.01 is default)
        pi.ac_noise_std=0.01
        env.env.collected_data = np.array([])
  
        rews, t, novelty_vector = pi.rollout(env, render=True, random_stream=np.random if stochastic else None)
        data_array = env.env.collected_data
        print('return={:.4f} len={}'.format(rews.sum(), t))

        #get filename from reward
        csvname = str(rews.sum()) + "_" + datetime.datetime.now().strftime("%d-%m-%y_%H-%M-%S") + "_results"

	#write csv
        with open('{}.csv'.format(csvname), 'w') as csvfile:
            fieldnames = ['speed', 'cost', 'c1','c2','c3','c4','c5','c6','c7','c8','c9','c10','c11','c12','c13','c14','c15','c16','c17','dist']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            steps = len(data_array)/20 #dividing by number of values in each substep of array 
            final_test = data_array
            for i in range(int(steps)):
                r = final_test[(i*20):((i+1)*20)] #speed, control values and distance travelled
                writer.writerow({'speed': r[0], 'cost': r[1], 'c1': r[2], 'c2': r[3], 'c3': r[4],'c4': r[5], 'c5': r[6],'c6': r[7],'c7': r[8],'c8': r[9],'c9': r[10],'c10': r[11],'c11': r[12],'c12': r[13],'c13': r[14],'c14': r[15],'c15': r[16],'c16': r[17],'c17': r[18], 'dist': r[19]})
 
#####################################   

        if record:
            env.close()
        return


if __name__ == '__main__':
    main()
