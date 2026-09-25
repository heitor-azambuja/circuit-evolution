#!/usr/bin/env bash
#
# Install and validate the simulation environment, for WSL/Ubuntu.
#
# Run it once per machine before starting a campaign. It fails loudly rather than
# leaving a half-working environment: every step that the campaign depends on is
# checked here, because discovering a missing shared library two hours into a run
# is expensive.
#
#   ./setup_env.sh
#
set -euo pipefail

cd "$(dirname "$0")"
VENV=venv

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
ok()   { printf '   \033[32mok\033[0m  %s\n' "$*"; }
fail() { printf '   \033[31mFALHOU\033[0m  %s\n' "$*" >&2; exit 1; }

say "Sistema"
[ -f /etc/debian_version ] || fail "esperado Debian/Ubuntu (WSL padrao); encontrado outro"
ok "$(. /etc/os-release && echo "$PRETTY_NAME")"
grep -qi microsoft /proc/version && ok "rodando sob WSL" || ok "nao e WSL (tudo bem)"

say "Pacotes do sistema"
# libngspice0 is the shared library PySpice loads; without the -dev package the
# loader cannot find libngspice.so, only the versioned file.
sudo apt-get update -qq
sudo apt-get install -y -qq ngspice libngspice0 libngspice0-dev \
                           python3-venv python3-dev build-essential
ok "ngspice $(ngspice -v 2>&1 | grep -oP 'ngspice-\d+' | head -1)"

say "Ambiente Python"
python3 -c 'import sys; sys.exit(0 if (3,9) <= sys.version_info < (3,13) else 1)' \
  || fail "Python $(python3 -V | cut -d' ' -f2) fora da faixa suportada (3.9 a 3.12)"
ok "$(python3 -V)"

[ -d "$VENV" ] || python3 -m venv "$VENV"
"$VENV"/bin/pip install -q --upgrade pip
# The lock pins what this project is known to run on. requirements.txt lists
# newer versions that have never been exercised here -- PySpice 1.5 predates
# numpy 2, which removed aliases it still uses.
"$VENV"/bin/pip install -q -r requirements-lock.txt
ok "dependencias instaladas a partir de requirements-lock.txt"

say "Validacao"
"$VENV"/bin/python - <<'PY' || fail "PySpice nao conseguiu simular"
import PySpice.Logging.Logging as Logging
Logging.setup_logging(logging_level='ERROR')
import numpy as np, circuits, filter_targets
f = circuits.SallenKeyLowPass()
f.configure_resistors([8107.5, 5259.8, 8581.5, 3599.7])
filter_targets.configure(f, 'butterworth_1000')
db = 20*np.log10(np.abs(np.array(f.ac_analysis()['out'])))
assert abs(db[40] + 3.0) < 0.2, f'resposta inesperada em 1 kHz: {db[40]:.3f} dB'
print(f'   ok  ngspice simulou o filtro: {db[40]:.3f} dB em 1 kHz (esperado ~-3)')
PY

PYTHONPATH=. "$VENV"/bin/python -m pytest tests/ -q >/dev/null 2>&1 \
  && ok "suite de testes passou" || fail "a suite de testes falhou"

say "Recursos e paralelismo sugerido"
CORES=$(nproc)
PHYS=$(lscpu -p=Core,Socket 2>/dev/null | grep -v '^#' | sort -u | wc -l)
MEM=$(free -g | awk '/^Mem:/{print $2}')
printf '   %s threads, %s nucleos fisicos, %s GB de RAM\n' "$CORES" "$PHYS" "$MEM"
# Each worker peaks near 0.6 GB because PySpice never returns what it leaks, so
# memory is rarely the limit; cores are.
SUGGEST=$(( PHYS + PHYS / 4 ))
[ "$SUGGEST" -gt "$CORES" ] && SUGGEST=$CORES
printf '   sugerido: \033[1m-w %s\033[0m  (memoria necessaria ~%s GB)\n' \
  "$SUGGEST" "$(( SUGGEST * 6 / 10 + 1 ))"
if grep -qi microsoft /proc/version && [ "$MEM" -lt 32 ]; then
  printf '   \033[33maviso\033[0m  o WSL2 limita a RAM por padrao; veja .wslconfig se precisar de mais\n'
fi

say "Tempo por execucao nesta maquina"
PYTHONPATH=. "$VENV"/bin/python - <<'PY'
import time, evolution_common as ec, filter_sk4_evolution as sk4
t = time.perf_counter()
run = ec.EvolutionRun(sk4.SPEC, target='butterworth_1000', exec_counter=0, seed=0)
run.evolve(40, 20, out_dir='/tmp/setup_check')   # 1/10 do orcamento real
dt = (time.perf_counter() - t) * 10
print(f'   ok  filtro 4R extrapolado para 400 geracoes: ~{dt:.0f}s por execucao')
print(f'       (referencia: ~40s numa Ryzen 5 2400G; menor e melhor)')
PY

printf '\n\033[1mAmbiente pronto.\033[0m Proximo passo: ./run_campaign.sh <i/N>\n\n'
