import pytest

import campaign

FAMILIES = ('filter', 'amp')


def _jobs(**kwargs):
    """Build the job list the way run() does, without running anything."""
    defaults = dict(family='filter', seeds=2, out_dir='out', patience=None,
                    population=None, variants=campaign.VARIANTS)
    defaults.update(kwargs)
    return [(defaults['family'], variant, target, seed, defaults['out_dir'],
             defaults['patience'], defaults['population'])
            for variant in defaults['variants']
            for target in campaign.targets_of(defaults['family'])
            for seed in range(1, defaults['seeds'] + 1)]


@pytest.mark.parametrize('family', FAMILIES)
def test_variant_selection_narrows_the_job_list_to_one_realization(family):
    both = _jobs(family=family)
    only_4r = _jobs(family=family, variants=('4R',))

    assert len(only_4r) * 2 == len(both)
    assert {job[1] for job in only_4r} == {'4R'}


@pytest.mark.parametrize('family', FAMILIES)
def test_population_override_reaches_every_job(family):
    jobs = _jobs(family=family, population=40)

    assert all(job[6] == 40 for job in jobs)


@pytest.mark.parametrize('family', FAMILIES)
def test_population_defaults_to_none_so_each_spec_keeps_its_own(family):
    """None, not a number: 4R and 8R take different populations from their specs."""
    jobs = _jobs(family=family)

    assert all(job[6] is None for job in jobs)


@pytest.mark.parametrize('family', FAMILIES)
def test_the_two_variants_ship_with_different_default_populations(family):
    """The reason --population exists. If this ever equalizes, the confound is gone.

    It is the same 20-vs-40 split in both families, so both need the same control.
    """
    populations = {variant: campaign._module(family, variant).SPEC.default_population
                   for variant in campaign.VARIANTS}

    assert populations['4R'] != populations['8R']


def _history(tmp_path, family, variant, seed=1):
    """Write a history file exactly where a run of that cell would put one."""
    spec = campaign._module(family, variant).SPEC
    slug = spec.evaluator.target_slug(campaign.targets_of(family)[0])
    path = tmp_path / f'fitness_history_{spec.circuit_name}_{slug}_execution{seed}.json'
    path.write_text('[]')
    return spec, path


@pytest.mark.parametrize('family', FAMILIES)
def test_history_collision_warning_is_silent_without_a_population_override(family, tmp_path):
    assert campaign.warn_about_history_collisions(family, ('4R',), str(tmp_path), None) == []


@pytest.mark.parametrize('family', FAMILIES)
def test_history_collision_warning_is_silent_at_the_spec_default(family, tmp_path):
    """At the default population there is no control run to keep separate."""
    spec, _ = _history(tmp_path, family, '4R')

    doomed = campaign.warn_about_history_collisions(
        family, ('4R',), str(tmp_path), spec.default_population)

    assert doomed == []


@pytest.mark.parametrize('family', FAMILIES)
def test_history_collision_warning_names_the_files_a_control_run_would_destroy(family, tmp_path):
    """The filename carries no population, so a control run silently overwrites.

    The warning is intentionally conservative: it cannot know what population wrote
    the files already there, only that this run will replace them.
    """
    spec, existing = _history(tmp_path, family, '4R', seed=7)

    doomed = campaign.warn_about_history_collisions(
        family, ('4R',), str(tmp_path), spec.default_population + 20)

    assert doomed == [str(existing)]


@pytest.mark.parametrize('family', FAMILIES)
def test_history_collision_warning_ignores_the_other_variant_s_files(family, tmp_path):
    """A 4R control must not report 8R histories, whose names differ."""
    _history(tmp_path, family, '8R')

    doomed = campaign.warn_about_history_collisions(family, ('4R',), str(tmp_path), 40)

    assert doomed == []


def test_history_collision_warning_keeps_the_families_apart(tmp_path):
    """Both families share an out-dir in run_campaign.sh, so an amp control must not
    count the filter's histories, nor the other way round."""
    _, amp_file = _history(tmp_path, 'amp', '4R')
    _, filter_file = _history(tmp_path, 'filter', '4R')

    assert campaign.warn_about_history_collisions(
        'amp', ('4R',), str(tmp_path), 40) == [str(amp_file)]
    assert campaign.warn_about_history_collisions(
        'filter', ('4R',), str(tmp_path), 40) == [str(filter_file)]
