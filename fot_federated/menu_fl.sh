#!/bin/bash

while true; do
  clear
  echo "🚀 MENU - Federated Learning Control Panel"
  echo "=========================================="
  echo "1. Iniciar sistema (start_fl_multi_full.sh)"
  echo "2. Parar sistema (stop_fl.sh)"
  echo "3. Reiniciar sistema (restart_fl.sh)"
  echo "4. Ver status dos containers (status_fl.sh)"
  echo "5. Salvar logs (logs_fl.sh)"
  echo "0. Sair"
  echo "=========================================="
  read -p "Escolha uma opção: " opcao

  case $opcao in
    1) bash start_fl_multi_full.sh ;;
    2) bash stop_fl.sh ;;
    3) bash restart_fl.sh ;;
    4) bash status_fl.sh ;;
    5) bash logs_fl.sh ;;
    0) echo "👋 Saindo..."; exit 0 ;;
    *) echo "Opção inválida. Pressione ENTER para continuar..."; read ;;
  esac

  echo
  read -p "Pressione ENTER para voltar ao menu..."
done
