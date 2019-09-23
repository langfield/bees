Call to ``PPOTrainer()`` constructor in ``bees/trainer.py`` begins on line 89 of ``trainer_template.py``.

In this ``__init__()`` function, we immediately call the init of ``Trainer`` in ``trainer.py``. This init function does nothing of interest until the end, when it calls the ``__init__()`` function of ``Trainable`` in ``ray/tune/trainable.py``.

This does nothing terribly interesting until line 96, where it calls ``self._setup()``.

This function is defined in ``trainer.py`` around line 412 (453 on github for some reason).

This is uninteresting until the line that reads:
    
    ```python
    self._init(self.config, self.env_creator)
    ```

around line 483 on github.

This brings us to ``trainer_template.py`` to the line around 92 that reads
    
    ```python
    def _init(self, config, env_creator):
    ```

Note that the following line executes because ``get_policy_class`` is None:

    ```python
    policy = default_policy
    ```

Note that ``default_policy`` is set to ``PPOTFPolicy``.

Note that the following lines execute because ``make_workers`` is None:

    ```python
    self.workers = self._make_workers(env_creator, policy, config,
                                                  self.config["num_workers"])
    ```

This calls ``_make_workers()`` in ``trainer.py``, which calls the ``WorkerSet`` constructor.

In the RolloutWorker init call, policy_map is built at line 345 ish in

        ```python
        if seed is not None:
            tf.set_random_seed(seed)
        self.policy_map, self.preprocessors = \
            self._build_policy_map(policy_dict, policy_config)
        ```

During worker creation, we are constructing ``RolloutWorker`` objects. In the init function of this class, near line 445, we call

    ```python
    self.input_reader = input_creator(self.io_context)
    ```

This simply sets ``self.input_reader`` equal to ``self.sampler`` from ``rollout_worker.py``, which in our case, is an instance of ``SyncSampler``. 

Then in ``sample()`` in ``RolloutWorker``, we call ``self.input_reader.next()``, in order to create ``batches``. This is ultimately calling the ``next()`` function from the ``SamplerInput`` class defined on line 55 in ``sampler.py``. Note that ``SyncSampler`` is a subclass of ``SamplerInput``, so when we call ``self.get_data()`` in this ``next()`` function, we are really calling ``get_data()`` as defined in ``SyncSampler``, which calls ``next(self.rollout_provider)``, which is just calling next of the generator returned by ``_env_runner()``, which is calling ``base_env.poll()`` to get observations, rewards, dones, and infos from the environment. Upon each ``next()`` call, we call ``_process_observations()`` on all of this data from the environment, and it is this function which is calling ``policy_for()``.

But note that the ``sample()`` function from ``RolloutWorker`` is ultimately being called within ``LocalMultiGPUOptimizer`` from ``multi_gpu_optimizer.py``. The ``LocalMultiGPUOptimizer`` class has a ``self.policies`` dictionary, which it assumes is static (it is constructed in the init function and then never modified). If we want to make sure everything is updated correctly, we're going to want to re-update this ``self.policies`` variable on every step by asking each worker how many policies it has. On the next level lower, we're going to want to just make the state of each ``RolloutWorker`` consistent.

If we change things, ``_build_policy_map()`` will have to change to be consistent with the way we will pass the policy objects through the policy mapping function during training.

Question: where are the policies actually instatiated? We need to know this because we need to instantiate more on every call to ``_process_observations``.

To answer this, note that in the ``RolloutWorker`` init function, we pass in ``policies`` and construct ``policy_dict`` which is a dictionary of policy id strings to (Policy, obs_space, action_space, config) tuples.

We set ``self.policies_to_train`` equal to just a list of the keys in the aforementioned dictionary. We then, somewhere around line 348 in ``rollout_worker.py``, call ``self._build_policy_map()``, which is defined in the same file. 

in  ``self._build_policy_map()``, we finally grab instances of the ``Policy`` class we've been passing around. This is done around line 764.

Note that ``self._build_policy_map()`` returns a state variable called ``self.policy_map``, which is used EVERYWHERE in ``RolloutWorker``, so we definitely need to be updating this. 

In addition, ``self.policy_map`` is being passed as the second nonself argument of ``SyncSampler``, which is called ``policies`` in ``SyncSampler``.  

Note the following line executes because ``make_policy_optimizer`` is defined as ``choose_policy_optimizer`` in ``ppo.py``. 

    ```python
    self.optimizer = make_policy_optimizer(self.workers, config)
    ```


===================

Train call.

The train call basically does a bunch of useless, roundabout inheritance and override calls until it gets to ``_train()`` in ``trainer_template.py`` around line 122. This is basically what is called every time we call ``train()`` in ``bees/trainer.py``.

This does nothing interesting until it calls ``self.optimizer.step()`` around line 129. In this case, our optimizer is ``MultiGPUOptimizer`` which is set in ``choose_policy_optimizer`` as described above.  
