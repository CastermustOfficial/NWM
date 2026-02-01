# NWM - Negative Weight Mapping

**Apprendimento per Esclusione** - Un framework che guida l'esplorazione e la stabilita  utilizzando memorie persistenti di fallimento e successo.

## Cos'e NWM?

Il **Negative Weight Mapping** trasforma l'esperienza passata in un **Campo di Forza Potenziale**. L'agente naviga lo spazio degli stati reagendo a:

- **Forze Attrattive** - Successi passati che guidano verso azioni ottimali
- **Forze Repulsive** - Fallimenti passati che allontanano da azioni pericolose

### Caratteristiche Principali

| Feature                        | Descrizione                                             |
| ------------------------------ | ------------------------------------------------------- |
| **Dynamic Smart Lock**   | Protegge automaticamente le memorie di alta qualita     |
| **Adaptive Exploration** | Riduce l'esplorazione quando trova strategie vincenti   |
| **Persistent Memory**    | I centroidi evolvono con "stiffness" progressiva        |
| **Fear & Greed**         | Evita le azioni pericolose prima di cercare il guadagno |

## Installazione

```bash
pip install nwm-rl
```

Oppure da sorgente:

```bash
git clone https://github.com/CastermustOfficial/NWM.git
cd NWM
pip install -e .
```

## Quick Start

```python
import gymnasium as gym
from nwm import NWM

# Crea ambiente e agente
env = gym.make("CartPole-v1")
agent = NWM(
    state_dim=env.observation_space.shape[0],
    num_actions=env.action_space.n
)

# Training loop
for episode in range(100):
    state, _ = env.reset()
    done = False
  
    while not done:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.step(state, action, reward, next_state, done)
        state = next_state
  
    print(f"Episode {episode + 1}: Best = {agent.best_reward:.0f}")
```

## API Reference

### NWM

```python
from nwm import NWM, NWMConfig

# Con configurazione personalizzata
config = NWMConfig(
    max_centroids=500,      # Massimo numero di centroidi
    warmup_episodes=50,     # Episodi di pura esplorazione
    exploration_rate=1.0,   # Tasso iniziale di esplorazione
    exploration_decay=0.99, # Decay dell'esplorazione
    min_exploration=0.05,   # Esplorazione minima
)

agent = NWM(state_dim=4, num_actions=2, config=config)

# Metodi principali
action = agent.select_action(state, training=True)
agent.step(state, action, reward, next_state, done)
stats = agent.get_stats()

# Save/Load
agent.save("agent.pkl")
agent = NWM.load("agent.pkl")
```

### NWMConfig

| Parametro             | Default | Descrizione                            |
| --------------------- | ------- | -------------------------------------- |
| `max_centroids`     | 500     | Numero massimo di centroidi in memoria |
| `warmup_episodes`   | 50      | Episodi di warmup prima del learning   |
| `exploration_rate`  | 1.0     | Tasso iniziale di esplorazione         |
| `exploration_decay` | 0.99    | Fattore di decay                       |
| `min_exploration`   | 0.05    | Esplorazione minima                    |
| `merge_threshold`   | 0.3     | Soglia per il merge dei centroidi      |
| `distance_cutoff`   | 2.5     | Distanza massima di influenza          |

## Struttura del Package

```
PYTHONLIB/
├── nwm/                    # Package principale
│   ├── agents/             # Implementazioni agenti
│   ├── core/               # Componenti core (centroid, potential_field)
│   └── utils/              # Utilities (config)
├── examples/               # Esempi di utilizzo
└── tests/                  # Test suite
```

## Esempi

```bash
# Quick start
python examples/quickstart.py

# Training completo
python examples/cartpole_training.py --episodes 500

# Demo visuale
python examples/cartpole_training.py --demo
```

---

**Versione**: 1.0.0
**Licenza**: MIT
**Autore**: CusterMustOfficial
