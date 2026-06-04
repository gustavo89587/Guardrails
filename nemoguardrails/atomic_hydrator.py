# Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Modulo de utilitarios para hidratacao e persistencia atomica de estado.
Provê isolamento de exclusao mutua distribuida por ID de conversa.
"""

import asyncio
from typing import Dict, Any, Callable

class AtomicStateHydrator:
    """
    Gerenciador de ciclo de vida assincrono para protecao de condicoes de corrida.
    Garante a linearizabilidade das operacoes de leitura e escrita de estado.
    """

    def __init__(self, backend_client: Any):
        """
        Inicializa o alicerce do hidratador atomico com o cliente de banco de dados.
        """
        self.backend_client = backend_client
        self._locks: Dict[str, asyncio.Lock] = {}
        self._ref_counts: Dict[str, int] = {}

    async def _acquire_session_lock(self, conversation_id: str) -> asyncio.Lock:
        """
        Acquire ou cria um sharded resource mutex exclusivo baseado no ID da conversa.
        """
        if conversation_id not in self._locks:
            self._locks[conversation_id] = asyncio.Lock()
            self._ref_counts[conversation_id] = 0
        self._ref_counts[conversation_id] += 1
        return self._locks[conversation_id]

    async def execute_atomic_pipeline(self, conversation_id: str, evaluation_coro: Callable) -> tuple:
        """
        Executa uma corrotina de avaliacao garantindo isolamento linearizavel de estado.

        Args:
            conversation_id (str): Identificador unico da sessao soberana.
            evaluation_coro (Callable): Logica de IA a ser julgada sob exclusao mutua.

        Returns:
            tuple: Par ordenado contendo os eventos de sada e o estado atualizado.
        """
        lock = await self._acquire_session_lock(conversation_id)
        async with lock:
            try:
                state = await self.backend_client.fetch_state(conversation_id)
                result, updated_state = await evaluation_coro(state)
                await self.backend_client.save_state(conversation_id, updated_state)
                return result, updated_state
            finally:
                self._ref_counts[conversation_id] -= 1
                if self._ref_counts[conversation_id] == 0:
                    del self._locks[conversation_id]
                    del self._ref_counts[conversation_id]
