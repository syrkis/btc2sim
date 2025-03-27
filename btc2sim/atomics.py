# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Import

# %%
from jax import vmap, random
from jax.lax import fori_loop
import jax.numpy as jnp
from flax.struct import dataclass

# %%
from btc2sim.dsl import unit_types
from btc2sim.bt import Parent

# %% [markdown]
# # Atomics

# %%
target_types = {
    "spearmen": 0,
    "archer": 1,
    "cavalry": 2,
    "healer": 3,
    "grenadier": 4,
    "any": None,
}


# %%
@dataclass
class Status:  # for behavior tree
    SUCCESS: int = 1
    FAILURE: int = -1
    NONE: int = 0


# %% [markdown]
# ## Action class

# %%
NONE, STAND, MOVE, ATTACK, HEAL, AREA_ATTACK = (
    jnp.array(-1),
    jnp.array(0),
    jnp.array(1),
    jnp.array(2),
    jnp.array(3),
    jnp.array(4),
)


# %%
@dataclass
class Action:
    kind: jnp.ndarray
    value: jnp.ndarray

    def __getitem__(self, index):  # to allow slicing operations
        return Action(
            kind=self.kind[index],
            value=self.value[index],
        )

    def set_item(self, index, new_value):
        # Perform an in-place update to kind and value at the specified index
        return Action(kind=self.kind.at[index].set(new_value.kind), value=self.value.at[index].set(new_value.value))

    def where(self, condition, false_value):
        return Action(
            kind=jnp.where(condition, self.kind, false_value.kind)[0],
            value=jnp.where(condition, self.value, false_value.value)[0],
        )

    @classmethod
    def from_shape(cls, shape, dtype=jnp.float32):
        # Create an instance with empty arrays of the specified shape
        return cls(kind=jnp.ones(shape, dtype=dtype) * NONE, value=jnp.zeros(shape + (2,), dtype=dtype))

    def conditional_action(self, condition: jnp.ndarray, action_if_true, action_if_false):
        return Action(
            kind=jnp.where(condition, action_if_true.kind, action_if_false.kind)[0],
            value=jnp.where(condition, action_if_true.value, action_if_false.value)[0],
        )


# %%
NONE_ACTION = Action(NONE, jnp.zeros((2,), dtype=jnp.float32))
STAND_ACTION = Action(STAND, jnp.zeros((2,), dtype=jnp.float32))


# %% [markdown]
# ## Miscellaneous


# %%
def has_line_of_sight(obstacles, source, target, env):
    # suppose that the target position is in sight_range of source, otherwise the line of sight might miss some cells
    current_line_of_sight = (
        source[:, jnp.newaxis] * (1 - env.scene.line_of_sight) + env.scene.line_of_sight * target[:, jnp.newaxis]
    )
    cells = jnp.array(current_line_of_sight, dtype=jnp.int32)
    in_sight = obstacles[cells[0], cells[1]].sum() == 0
    return in_sight


# %%
def stand_factory(all_variants):
    def stand(env, scenario, state, agent_id, variants_status, variants_action):
        variant_id = all_variants.index("stand")
        return variants_status.at[variant_id].set(Status.SUCCESS), variants_action.set_item(variant_id, STAND_ACTION)

    return stand


# %% [markdown]
def debug_factory(all_variants):
    def debug(env, scenario, state, agent_id, variants_status, variants_action):
        def aux(variant_id, motion):
            return variants_status.at[variant_id].set(Status.SUCCESS), variants_action.set_item(
                variant_id, Action(MOVE, motion)
            )

        velocity = env.unit_type_velocities[scenario.unit_type[agent_id]]
        variants_status, variants_action = aux(all_variants.index("debug north"), jnp.array([0, velocity]))  # type: ignore
        variants_status, variants_action = aux(all_variants.index("debug south"), jnp.array([0, -velocity]))  # type: ignore
        variants_status, variants_action = aux(all_variants.index("debug east"), jnp.array([velocity, 0]))  # type: ignore
        variants_status, variants_action = aux(all_variants.index("debug west"), jnp.array([-velocity, 0]))  # type: ignore
        return variants_status, variants_action

    return debug


# %% [markdown]
# ### Attack


# %%
def attack_factory(all_variants):
    def attack(env, scenario, state, obs, rng, agent_id, variants_status, variants_action):
        can_attack = jnp.logical_and(
            state.unit_cooldown[agent_id] <= 0, scenario.unit_type[agent_id] != target_types["healer"]
        )
        attack_type = jnp.where(scenario.unit_type[agent_id] == target_types["grenadier"], AREA_ATTACK, ATTACK)
        close_dist_matrix = obs.dist[agent_id]
        close_dist_matrix = jnp.where(
            scenario.unit_team != scenario.unit_team[agent_id], close_dist_matrix, jnp.inf
        )  # only enemies
        close_dist_matrix = jnp.where(  # type:ignore
            close_dist_matrix <= env.scene.unit_type_attack_ranges[scenario.unit_type[agent_id]],
            close_dist_matrix,  # type:ignore
            jnp.inf,  # type:ignore
        )  # in attack range # type:ignore
        far_dist_matrix = jnp.where(close_dist_matrix == jnp.inf, -jnp.inf, close_dist_matrix)  # we want to use argmax

        def aux(value, extremum, variant, variants_status, variants_action):
            target_id = (jnp.argmax if extremum == "max" else jnp.argmin)(value)
            variant_id = all_variants.index("attack " + variant)
            flag = jnp.where(
                value[target_id] != (jnp.inf if extremum == "min" else -jnp.inf), Status.SUCCESS, Status.FAILURE
            )
            flag = jnp.where(can_attack, flag, Status.FAILURE)
            action = Action(
                kind=jnp.where(flag == Status.SUCCESS, attack_type, NONE),
                value=jnp.where(flag == Status.SUCCESS, jnp.array([target_id, 0], dtype=jnp.float32), jnp.zeros(2)),
            )
            variants_status = variants_status.at[variant_id].set(flag)
            variants_action = variants_action.set_item(variant_id, action)
            return variants_status, variants_action

        # random
        value = jnp.where(close_dist_matrix != jnp.inf, 1, -jnp.inf)
        value += random.uniform(rng, value.shape) * 0.5
        variants_status, variants_action = aux(value, "max", "random any", variants_status, variants_action)
        # distances
        variants_status, variants_action = aux(
            close_dist_matrix, "min", "closest any", variants_status, variants_action
        )
        variants_status, variants_action = aux(far_dist_matrix, "max", "farthest any", variants_status, variants_action)
        # health
        min_health = jnp.where(close_dist_matrix != jnp.inf, state.unit_health, jnp.inf)
        min_health += random.uniform(rng, min_health.shape) * 0.5  # type: ignore
        max_health = jnp.where(min_health == jnp.inf, -jnp.inf, min_health)
        variants_status, variants_action = aux(min_health, "min", "weakest any", variants_status, variants_action)
        variants_status, variants_action = aux(max_health, "max", "strongest any", variants_status, variants_action)

        for unit_type in unit_types:
            units = scenario.unit_type == target_types[unit_type]
            # random
            variants_status, variants_action = aux(
                jnp.where(units, value, -jnp.inf), "max", f"random {unit_type}", variants_status, variants_action
            )
            # distances
            variants_status, variants_action = aux(
                jnp.where(units, close_dist_matrix, jnp.inf),
                "min",
                f"closest {unit_type}",
                variants_status,
                variants_action,
            )
            variants_status, variants_action = aux(
                jnp.where(units, far_dist_matrix, -jnp.inf),  # type: ignore
                "max",
                f"farthest {unit_type}",
                variants_status,
                variants_action,
            )
            # health
            variants_status, variants_action = aux(
                jnp.where(units, min_health, jnp.inf), "min", f"weakest {unit_type}", variants_status, variants_action
            )
            variants_status, variants_action = aux(
                jnp.where(units, max_health, -jnp.inf),
                "max",
                f"strongest {unit_type}",
                variants_status,
                variants_action,
            )

        return variants_status, variants_action

    return attack


# %% [markdown]
# ### Move


# %%
def move_factory(all_variants):
    def move(env, scenario, state, obs, rng, agent_id, variants_status, variants_action):
        close_dist_matrix = obs.dist
        far_dist_matrix = jnp.where(close_dist_matrix == jnp.inf, -jnp.inf, close_dist_matrix)  # we want to use argmax
        foes = scenario.unit_team != scenario.unit_team[agent_id]
        friends = scenario.unit_team == scenario.unit_team[agent_id]

        def aux(value, extremum, variant, variants_status, variants_action):
            target_id = (jnp.argmax if extremum == "max" else jnp.argmin)(value)
            valid_target = value[target_id] != (jnp.inf if extremum == "min" else -jnp.inf)
            variant_id_toward, variant_id_away_from = (
                all_variants.index("move toward " + variant),
                all_variants.index("move away_from " + variant),
            )
            delta = state.unit_position[target_id] - state.unit_position[agent_id]
            norm = jnp.linalg.norm(delta)
            velocity = env.cfg.unit_type_velocities[scenario.unit_type[agent_id]]
            delta = jnp.where(norm <= velocity, delta, velocity * delta / norm)
            obstacles = scenario.terrain.building + scenario.terrain.water  # cannot cross building and water

            can_move_toward_target = has_line_of_sight(
                obstacles, state.unit_position[agent_id], state.unit_position[agent_id] + delta, env
            )
            can_move_away_from_target = has_line_of_sight(
                obstacles, state.unit_position[agent_id], state.unit_position[agent_id] - delta, env
            )

            flag_toward = jnp.where(
                jnp.logical_and(can_move_toward_target, valid_target), Status.SUCCESS, Status.FAILURE
            )
            flag_away_from = jnp.where(
                jnp.logical_and(can_move_away_from_target, valid_target), Status.SUCCESS, Status.FAILURE
            )
            action_toward = Action(
                kind=jnp.where(flag_toward == Status.SUCCESS, MOVE, NONE),
                value=jnp.where(flag_toward == Status.SUCCESS, delta, jnp.zeros(2))[0],
            )
            action_away_from = Action(
                kind=jnp.where(flag_away_from == Status.SUCCESS, MOVE, NONE),
                value=jnp.where(flag_away_from == Status.SUCCESS, -delta, jnp.zeros(2))[0],
            )

            variants_status = variants_status.at[variant_id_toward].set(flag_toward)
            variants_status = variants_status.at[variant_id_away_from].set(flag_away_from)
            variants_action = variants_action.set_item(variant_id_toward, action_toward)
            variants_action = variants_action.set_item(variant_id_away_from, action_away_from)
            return variants_status, variants_action

        # random
        value = jnp.where(close_dist_matrix != jnp.inf, 1, -jnp.inf)
        value += random.uniform(rng, value.shape) * 0.5
        # health
        min_health = jnp.where(close_dist_matrix != jnp.inf, state.unit_health, jnp.inf)
        min_health += random.uniform(rng, min_health.shape) * 0.5  # type: ignore
        max_health = jnp.where(min_health == jnp.inf, -jnp.inf, min_health)

        for team_name, team in zip(["friend", "foe"], [friends, foes]):
            # random
            variants_status, variants_action = aux(
                jnp.where(team, value, -jnp.inf), "max", f"random {team_name} any", variants_status, variants_action
            )
            # distance
            variants_status, variants_action = aux(
                jnp.where(team, close_dist_matrix, jnp.inf),
                "min",
                f"closest {team_name} any",
                variants_status,
                variants_action,
            )
            variants_status, variants_action = aux(
                jnp.where(team, far_dist_matrix, -jnp.inf),  # type: ignore
                "max",
                f"farthest {team_name} any",
                variants_status,
                variants_action,
            )
            # health
            variants_status, variants_action = aux(
                jnp.where(team, min_health, jnp.inf),
                "min",
                f"weakest {team_name} any",
                variants_status,
                variants_action,
            )
            variants_status, variants_action = aux(
                jnp.where(team, max_health, -jnp.inf),
                "max",
                f"strongest {team_name} any",
                variants_status,
                variants_action,
            )
            for unit_type in unit_types:
                units = jnp.logical_and(scenario.unit_type == target_types[unit_type], team)
                # random
                variants_status, variants_action = aux(
                    jnp.where(team, value, -jnp.inf),
                    "max",
                    f"random {team_name} {unit_type}",
                    variants_status,
                    variants_action,
                )
                # distance
                variants_status, variants_action = aux(
                    jnp.where(units, close_dist_matrix, jnp.inf),
                    "min",
                    f"closest {team_name} {unit_type}",
                    variants_status,
                    variants_action,
                )
                variants_status, variants_action = aux(
                    jnp.where(units, far_dist_matrix, -jnp.inf),  # type: ignore
                    "max",
                    f"farthest {team_name} {unit_type}",
                    variants_status,
                    variants_action,
                )
                # health
                variants_status, variants_action = aux(
                    jnp.where(units, min_health, jnp.inf),
                    "min",
                    f"weakest {team_name} {unit_type}",
                    variants_status,
                    variants_action,
                )
                variants_status, variants_action = aux(
                    jnp.where(units, max_health, -jnp.inf),
                    "max",
                    f"strongest {team_name} {unit_type}",
                    variants_status,
                    variants_action,
                )

        return variants_status, variants_action

    return move


# %% [markdown]
# ### Follow map


# %%
def follow_map_factory(all_variants):
    n_direction = 8  # number of direction arround the unit (2pi/n_direction)
    n_step_size = 4  # number of steps in the direction up to the unit's velocity (should be at least equal to the max velocity so that it check every cells

    def follow_map(env, scene, state, rng, agent_id, variants_status, variants_action):
        candidates = jnp.array(
            [[0, 0]]
            + [
                [
                    step_size / n_step_size * jnp.cos(2 * jnp.pi * theta / n_direction),
                    step_size / n_step_size * jnp.sin(2 * jnp.pi * theta / n_direction),
                ]
                for theta in jnp.arange(n_direction)
                for step_size in jnp.arange(1, n_step_size + 1)
            ]
        )
        candidates *= scene.unit_type_velocities[scene.unit_type[agent_id]]
        candidates_idx = jnp.array(state.unit_position[agent_id] + candidates, dtype=jnp.uint32)
        candidates_idx = jnp.clip(candidates_idx, 0, env.cfg.size - 1)

        distances = scene.distance_map[scene.unit_target_position_id[agent_id]][
            candidates_idx[:, 0], candidates_idx[:, 1]
        ]
        distances += random.uniform(
            rng, distances.shape, minval=0.0, maxval=scene.movement_randomness
        )  # to resolve tighs and give a more organic vibe
        obstacles = scene.terrain.building + scene.terrain.water  # cannot walk through building and water
        in_sight = vmap(has_line_of_sight, in_axes=(None, None, 0, None))(
            obstacles, state.unit_position[agent_id], state.unit_position[agent_id] + candidates, env
        )
        distances_toward = jnp.where(in_sight, distances, env.size**2)  # in sight positions
        distances_away = jnp.where(distances_toward >= env.size**2, -1, distances_toward)

        for margin_name, margin in zip(["0%", "25%", "50%", "100%"], [0, 0.25, 0.5, 1.0]):
            d = env.unit_type_sight_ranges[scene.unit_type[agent_id]] * margin
            for sense_name, target_idx in zip(
                ["toward", "away_from"],
                [jnp.argmin(distances_toward), jnp.argmax(distances_away)],  # type: ignore
            ):
                if margin_name == "0%" and sense_name == "away_from":
                    d = jnp.inf
                else:
                    d *= margin
                margin_check = distances_toward[0] > d if sense_name == "toward" else distances_away[0] < d
                need_to_move = target_idx != 0  # atomic return Failure if the agent doesn't need to move or can't move
                flag = jnp.where(jnp.logical_and(margin_check, need_to_move), Status.SUCCESS, Status.FAILURE)
                action = Action(kind=jnp.where(flag == Status.SUCCESS, MOVE, NONE), value=candidates[target_idx])
                variant_id = all_variants.index(f"follow_map {sense_name} {margin_name}")
                variants_status, variants_action = (
                    variants_status.at[variant_id].set(flag),
                    variants_action.set_item(variant_id, action),
                )

        return variants_status, variants_action

    return follow_map


# %% [markdown]
# ### Heal


# %%
def heal_factory(all_variants):
    def heal(env, scenario, state, rng, agent_id, variants_status, variants_action):
        can_heal = jnp.logical_and(
            state.unit_cooldown[agent_id] <= 0, scenario.unit_type[agent_id] == target_types["healer"]
        )
        close_dist_matrix = state.unit_in_sight_distance[agent_id]
        close_dist_matrix = jnp.where(
            scenario.unit_team == scenario.unit_team[agent_id], close_dist_matrix, jnp.inf
        )  # only allies
        # in attack range
        close_dist_matrix = jnp.where(  # type: ignore
            close_dist_matrix <= env.unit_type_attack_ranges[scenario.unit_type[agent_id]],
            close_dist_matrix,  # type: ignore
            jnp.inf,  # type: ignore
        )  # type: ignore
        far_dist_matrix = jnp.where(close_dist_matrix == jnp.inf, -jnp.inf, close_dist_matrix)  # we want to use argmax

        def aux(value, extremum, variant, variants_status, variants_action):
            target_id = (jnp.argmax if extremum == "max" else jnp.argmin)(value)
            variant_id = all_variants.index("heal " + variant)
            flag = jnp.where(
                value[target_id] != (jnp.inf if extremum == "min" else -jnp.inf), Status.SUCCESS, Status.FAILURE
            )
            flag = jnp.where(can_heal, flag, Status.FAILURE)
            action = Action(
                kind=jnp.where(flag == Status.SUCCESS, HEAL, NONE),
                value=jnp.where(flag == Status.SUCCESS, jnp.array([target_id, 0], dtype=jnp.float32), jnp.zeros(2)),
            )
            variants_status = variants_status.at[variant_id].set(flag)
            variants_action = variants_action.set_item(variant_id, action)
            return variants_status, variants_action

        # random
        value = jnp.where(close_dist_matrix != jnp.inf, 1, -jnp.inf)
        value += random.uniform(rng, value.shape) * 0.5
        variants_status, variants_action = aux(value, "max", "random any", variants_status, variants_action)
        # distances
        variants_status, variants_action = aux(
            close_dist_matrix, "min", "closest any", variants_status, variants_action
        )
        variants_status, variants_action = aux(far_dist_matrix, "max", "farthest any", variants_status, variants_action)
        # health
        min_health = jnp.where(close_dist_matrix != jnp.inf, state.unit_health, jnp.inf)
        min_health += random.uniform(rng, min_health.shape) * 0.5  # type: ignore
        max_health = jnp.where(min_health == jnp.inf, -jnp.inf, min_health)
        variants_status, variants_action = aux(min_health, "min", "weakest any", variants_status, variants_action)
        variants_status, variants_action = aux(max_health, "max", "strongest any", variants_status, variants_action)

        for unit_type in unit_types:
            units = scenario.unit_type == target_types[unit_type]
            # random
            variants_status, variants_action = aux(
                jnp.where(units, value, -jnp.inf), "max", f"random {unit_type}", variants_status, variants_action
            )
            # distances
            variants_status, variants_action = aux(
                jnp.where(units, close_dist_matrix, jnp.inf),
                "min",
                f"closest {unit_type}",
                variants_status,
                variants_action,
            )
            variants_status, variants_action = aux(
                jnp.where(units, far_dist_matrix, -jnp.inf),  # type: ignore
                "max",
                f"farthest {unit_type}",
                variants_status,
                variants_action,
            )
            # health
            variants_status, variants_action = aux(
                jnp.where(units, min_health, jnp.inf), "min", f"weakest {unit_type}", variants_status, variants_action
            )
            variants_status, variants_action = aux(
                jnp.where(units, max_health, -jnp.inf),
                "max",
                f"strongest {unit_type}",
                variants_status,
                variants_action,
            )

        return variants_status, variants_action

    return heal


# %% [markdown]
# ## Conditions atomics

# %% [markdown]
# ### In Sight

# %% [raw]
# in_sight  : "in_sight" (foe | friend) (unit | any)


# %%
def in_sight_factory(all_variants, n_agents):
    def in_sight(env, scenario, state, agent_id, variants_status):
        dist_matrix = state.unit_in_sight_distance[agent_id]
        foes = scenario.unit_team != scenario.unit_team[agent_id]
        friends = scenario.unit_team == scenario.unit_team[agent_id]
        any_unit = jnp.ones(n_agents, dtype=jnp.bool)

        def aux(target, unit_type):
            return jnp.where(
                jnp.any(jnp.where(jnp.logical_and(target, unit_type), dist_matrix, jnp.inf) != jnp.inf),
                Status.SUCCESS,
                Status.FAILURE,
            )

        for team_name, team in zip(["friend", "foe"], [friends, foes]):
            variants_status = variants_status.at[all_variants.index(f"in_sight {team_name} any")].set(
                aux(team, any_unit)
            )
            for unit_type in unit_types:
                units = scenario.unit_type == target_types[unit_type]
                variants_status = variants_status.at[all_variants.index(f"in_sight {team_name} {unit_type}")].set(
                    aux(team, units)
                )
        return variants_status

    return in_sight


# %% [markdown]
# ### In Reach


# %%
def in_reach_factory(all_variants, n_agents):
    def in_reach(env, scenario, state, agent_id, variants_status):
        foes = scenario.unit_team != scenario.unit_team[agent_id]
        friends = scenario.unit_team == scenario.unit_team[agent_id]
        spearmen = scenario.unit_type == target_types["spearmen"]
        archer = scenario.unit_type == target_types["archer"]
        cavalry = scenario.unit_type == target_types["cavalry"]
        healer = scenario.unit_type == target_types["healer"]
        grenadier = scenario.unit_type == target_types["grenadier"]

        any_unit = jnp.ones(n_agents, dtype=jnp.bool)
        dist_matrix = state.unit_in_sight_distance[agent_id]
        in_reach_0_from_me = dist_matrix <= env.unit_type_attack_ranges[scenario.unit_type[agent_id]]
        in_reach_1_from_me = dist_matrix <= (
            env.unit_type_attack_ranges[scenario.unit_type[agent_id]]
            + env.unit_type_velocities[scenario.unit_type[agent_id]]
        )
        in_reach_2_from_me = dist_matrix <= (
            env.unit_type_attack_ranges[scenario.unit_type[agent_id]]
            + 2 * env.unit_type_velocities[scenario.unit_type[agent_id]]
        )
        in_reach_3_from_me = dist_matrix <= (
            env.unit_type_attack_ranges[scenario.unit_type[agent_id]]
            + 3 * env.unit_type_velocities[scenario.unit_type[agent_id]]
        )
        them_from_me = [in_reach_0_from_me, in_reach_1_from_me, in_reach_2_from_me, in_reach_3_from_me]
        in_reach_0_from_them = dist_matrix <= env.unit_type_attack_ranges[scenario.unit_type]
        in_reach_1_from_them = dist_matrix <= (
            env.unit_type_attack_ranges[scenario.unit_type] + env.unit_type_velocities[scenario.unit_type]
        )
        in_reach_2_from_them = dist_matrix <= (
            env.unit_type_attack_ranges[scenario.unit_type] + 2 * env.unit_type_velocities[scenario.unit_type]
        )
        in_reach_3_from_them = dist_matrix <= (
            env.unit_type_attack_ranges[scenario.unit_type] + 3 * env.unit_type_velocities[scenario.unit_type]
        )
        me_from_them = [in_reach_0_from_them, in_reach_1_from_them, in_reach_2_from_them, in_reach_3_from_them]
        for source_name, sources in zip(["me_from_them", "them_from_me"], [me_from_them, them_from_me]):
            for type_name, unit_type in zip(
                ["spearmen", "archer", "cavalry", "healer", "grenadier", "any"],
                [spearmen, archer, cavalry, healer, grenadier, any_unit],
            ):
                for team_name, unit_team in zip(["foe", "friend"], [foes, friends]):
                    for steps_name, in_reach_steps in zip(["0", "1", "2", "3"], sources):
                        variant_id = all_variants.index(f"in_reach {team_name} {source_name} {steps_name} {type_name}")
                        variants_status = variants_status.at[variant_id].set(
                            jnp.where(
                                jnp.any(jnp.logical_and(jnp.logical_and(in_reach_steps, unit_type), unit_team)),
                                Status.SUCCESS,
                                Status.FAILURE,
                            )
                        )
        return variants_status

    return in_reach


# %% [markdown]
# ### Is_Type


# %%
def is_type_factory(all_variants):
    def is_type(env, scenario, state, agent_id, variants_status):
        for unit_type in unit_types:
            variants_status = variants_status.at[all_variants.index(f"is_type {unit_type}")].set(
                jnp.where(scenario.unit_type[agent_id] == target_types[unit_type], Status.SUCCESS, Status.FAILURE)
            )
        return variants_status

    return is_type


# %% [markdown]
# ### Is Dying


# %%
def is_dying_factory(all_variants):
    def is_dying(env, scenario, state, agent_id, variants_status):
        dist_matrix = state.unit_in_sight_distance[agent_id]
        foes = jnp.where(scenario.unit_team != scenario.unit_team[agent_id], dist_matrix, jnp.inf)
        friends = jnp.where(scenario.unit_team == scenario.unit_team[agent_id], dist_matrix, jnp.inf)
        for threshold_name, threshold in zip(["25%", "50%", "75%"], [0.25, 0.5, 0.75]):
            flag = jnp.where(
                state.unit_health[agent_id] / env.unit_type_health[scenario.unit_type[agent_id]] <= threshold,
                Status.SUCCESS,
                Status.FAILURE,
            )
            variants_status = variants_status.at[all_variants.index(f"is_dying self {threshold_name}")].set(flag)
            foes_flag = jnp.where(
                jnp.any(
                    jnp.where(foes != jnp.inf, state.unit_health / env.unit_type_health[scenario.unit_type], 1)[0]
                    <= threshold
                ),
                Status.SUCCESS,
                Status.FAILURE,
            )
            variants_status = variants_status.at[all_variants.index(f"is_dying foe {threshold_name}")].set(foes_flag)
            friends_flag = jnp.where(
                jnp.any(
                    jnp.where(friends != jnp.inf, state.unit_health / env.unit_type_health[scenario.unit_type], 1)[0]
                    <= threshold
                ),
                Status.SUCCESS,
                Status.FAILURE,
            )
            variants_status = variants_status.at[all_variants.index(f"is_dying friend {threshold_name}")].set(
                friends_flag
            )
        return variants_status

    return is_dying


# %% [markdown]
# ### Is in forest


# %%
def is_in_forest_factory(all_variants):
    def is_in_forest(env, scenario, state, agent_id, variants_status):
        pos = state.unit_position[agent_id].astype(jnp.uint32)
        variants_status = variants_status.at[all_variants.index("is_in_forest")].set(
            jnp.where(scenario.terrain.forest[pos[0], pos[1]], Status.SUCCESS, Status.FAILURE)
        )
        return variants_status

    return is_in_forest


# %% [markdown]
# ## Compute all variants


# %%
def compute_variants_factory(all_variants, n_agents):
    stand_eval = stand_factory(all_variants)
    move_eval = move_factory(all_variants)
    attack_eval = attack_factory(all_variants)
    follow_map_eval = follow_map_factory(all_variants)
    heal_eval = heal_factory(all_variants)
    # debug_eval = debug_factory(all_variants)
    in_sight_eval = in_sight_factory(all_variants, n_agents)
    in_reach_eval = in_reach_factory(all_variants, n_agents)
    is_type_eval = is_type_factory(all_variants)
    is_dying_eval = is_dying_factory(all_variants)
    is_in_forest_eval = is_in_forest_factory(all_variants)

    def compute_variants(env, state, obs, rng, agent_id, variants_status, variants_action):
        move_rng, attack_rng, follow_map_rng, heal_rng = random.split(rng, 4)
        variants_status, variants_action = stand_eval(env, env.scene, state, agent_id, variants_status, variants_action)
        variants_status, variants_action = move_eval(
            env, env.scene, state, obs, move_rng, agent_id, variants_status, variants_action
        )
        variants_status, variants_action = attack_eval(
            env, env.scene, state, obs, attack_rng, agent_id, variants_status, variants_action
        )
        variants_status, variants_action = follow_map_eval(
            env, env.scene, state, follow_map_rng, agent_id, variants_status, variants_action
        )
        variants_status, variants_action = heal_eval(
            env, env.scene, state, heal_rng, agent_id, variants_status, variants_action
        )
        # variants_status, variants_action = debug_eval(env, scenario, state, agent_id, variants_status, variants_action)
        variants_status = in_sight_eval(env, env.scene, state, agent_id, variants_status)
        variants_status = in_reach_eval(env, env.scene, state, agent_id, variants_status)
        variants_status = is_type_eval(env, env.scene, state, agent_id, variants_status)
        variants_status = is_dying_eval(env, env.scene, state, agent_id, variants_status)
        variants_status = is_in_forest_eval(env, env.scene, state, agent_id, variants_status)
        return variants_status, variants_action

    return compute_variants


# %% [markdown]
# # get action


# %%
def eval_bt(predecessors, parents, passing_nodes, variant_ids, variants_status, variants_action):
    def eval_leaf(i, carry):
        s, a, passing = carry
        variant_id = variant_ids[i]
        has_not_found_action = jnp.logical_or(s != Status.SUCCESS, a.kind == NONE)
        is_valid_from_sequence = jnp.logical_and(predecessors[i] == Parent.SEQUENCE, s != Status.FAILURE)
        is_valid_from_fallback = jnp.logical_and(predecessors[i] == Parent.FALLBACK, s != Status.SUCCESS)
        is_valid_from_root = predecessors[i] == Parent.NONE

        is_valid = jnp.logical_or(jnp.logical_or(is_valid_from_sequence, is_valid_from_fallback), is_valid_from_root)
        is_valid = jnp.logical_and(is_valid, variant_id != -1)  # not an empty leaf (from fixed size)
        condition = jnp.logical_and(jnp.logical_and(has_not_found_action, is_valid), passing <= 0)

        passing_if_FAILURE_in_sequence = jnp.logical_and(
            parents[i] == Parent.SEQUENCE, variants_status[variant_id] == Status.FAILURE
        )
        passing_if_SUCCESS_in_failure = jnp.logical_and(
            parents[i] == Parent.FALLBACK, variants_status[variant_id] == Status.SUCCESS
        )

        if_passing = jnp.logical_and(
            condition, jnp.logical_or(passing_if_FAILURE_in_sequence, passing_if_SUCCESS_in_failure)
        )
        passing = jnp.where(if_passing, passing_nodes[i], passing - 1)

        s = jnp.where(condition, variants_status[variant_id], s)
        a = variants_action[variant_id].where(condition, a)
        return s, a, passing

    return eval_leaf


# %%
def make_action_fn(all_variants, n_agents, bt_max_size):
    n_variants = len(all_variants)
    compute_variants = compute_variants_factory(all_variants, n_agents)

    def get_action(env, rng, state, obs, behavior, agent_id):  # for one agent
        variants_status, variants_action = compute_variants(
            env, state, obs, rng, agent_id, jnp.zeros(n_variants), Action.from_shape((n_variants,))
        )
        eval_leaf = eval_bt(
            behavior.predecessors,
            behavior.parents,
            behavior.passing_nodes,
            behavior.variant_ids,
            variants_status,
            variants_action,
        )
        carry = Status.NONE, NONE_ACTION, 0
        s, a, p = fori_loop(0, bt_max_size, eval_leaf, carry)
        return a.where(jnp.logical_and(s == Status.SUCCESS, a.kind != NONE), STAND_ACTION)

    return get_action
