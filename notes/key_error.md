<!--
General questions should be asked on the mailing list ray-dev@googlegroups.com.
Questions about how to use Ray should be asked on
[StackOverflow](https://stackoverflow.com/questions/tagged/ray).

Before submitting an issue, please fill out the following form.
-->

### System information
- **OS Platform and Distribution (e.g., Linux Ubuntu 16.04)**: Linux Ubuntu 18.04
- **Ray installed from (source or binary)**: Binary
- **Ray version**: 0.7.4
- **Python version**: 3.7.4
- **Exact command to reproduce**: N/A

<!--
You can obtain the Ray version with

python -c "import ray; print(ray.__version__)"
-->

### Describe the problem
<!-- Describe the problem clearly here. -->
I'm curious about the bit of code below from `sampler.py`. Why are all policies added to `to_eval` at the end of an episode, whereas on ordinary iterations, only policies for which the agent is not `done` get their `to_eval` lists appended to (see the second snippet). 

I ask because in #5753, I'm attempting to efficient initialization and termination of agents/policies during training. I'm removing agents from the `policies` dict passed around in `sampler.py` as agents return `done` to avoid overhead incurred from looping over a large list of `done` agents. Unfortunately, I'm getting funky fresh key errors as a result of `to_eval` being full of all the dead policy ids. 

### Source code / logs
<!-- Include any logs or source code that would be helpful to diagnose the problem. If including tracebacks, please include the full traceback. Large logs and files should be attached. Try to provide a reproducible test case that is the bare minimum necessary to generate the problem. -->

`Line 620`
```python
            elif resetted_obs != ASYNC_RESET_RETURN:
                print("Executing new epsiode non-async return.")
                time.sleep(1)
                raise NotImplementedError("Multiple episodes not supported by design.")
                # Creates a new episode if this is not async return
                # If reset is async, we will get its result in some future poll
                episode = active_episodes[env_id]
                for agent_id, raw_obs in resetted_obs.items():

                    #===MOD===
                    policy_id, policy_constructor_tuple = episode.policy_for(agent_id)
                    # with tf_sess.as_default():
                    pols_tuple = generate_policies(
                        policy_id,
                        policy_constructor_tuple,
                        policies,
                        policies_to_train,
                        dead_policies,
                        policy_config,
                        preprocessors,
                        obs_filters,
                        observation_filter,
                        tf_sess,
                    )
                    policies, preprocessors, obs_filters, policies_to_train, dead_policies = pols_tuple
                    #===MOD===

                    policy = _get_or_raise(policies, policy_id)
                    prep_obs = _get_or_raise(preprocessors,
                                             policy_id).transform(raw_obs)
                    filtered_obs = _get_or_raise(obs_filters,
                                                 policy_id)(prep_obs)
                    episode._set_last_observation(agent_id, filtered_obs)
                    to_eval[policy_id].append(
                        PolicyEvalData(
                            env_id, agent_id, filtered_obs,
                            episode.last_info_for(agent_id) or {},
                            episode.rnn_state_for(agent_id),
                            np.zeros_like(
                                _flatten_action(policy.action_space.sample())),
                            0.0))
```

`Line 536`
```python
        # For each agent in the environment
        for agent_id, raw_obs in agent_obs.items():

            #===MOD===
            policy_id, policy_constructor_tuple = episode.policy_for(agent_id)
            pols_tuple = generate_policies(
                policy_id,
                policy_constructor_tuple,
                policies,
                policies_to_train,
                dead_policies,
                policy_config,
                preprocessors,
                obs_filters,
                observation_filter,
                tf_sess,
            )
            policies, preprocessors, obs_filters, policies_to_train, dead_policies = pols_tuple
            #===MOD===

            prep_obs = _get_or_raise(preprocessors,
                                     policy_id).transform(raw_obs)
            if log_once("prep_obs"):
                logger.info("Preprocessed obs: {}".format(summarize(prep_obs)))

            filtered_obs = _get_or_raise(obs_filters, policy_id)(prep_obs)
            if log_once("filtered_obs"):
                logger.info("Filtered obs: {}".format(summarize(filtered_obs)))

            agent_done = bool(all_done or dones[env_id].get(agent_id))
            if not agent_done:
                to_eval[policy_id].append(
                    PolicyEvalData(env_id, agent_id, filtered_obs,
                                   infos[env_id].get(agent_id, {}),
                                   episode.rnn_state_for(agent_id),
                                   episode.last_action_for(agent_id),
                                   rewards[env_id][agent_id] or 0.0))
            #===MOD===
            else:
                # Does it make sense to remove agent id from `agent_builders`?
                dead_policies.add(policy_id)
                episode.batch_builder.agent_builders.pop(agent_id)
                if policy_id in to_eval:
                    to_eval.pop(policy_id)
                    print("Popping policy id from toeval.")
                    time.sleep(1)
            #===MOD===
```

