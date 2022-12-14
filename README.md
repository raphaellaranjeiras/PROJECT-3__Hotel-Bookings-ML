![](hotel.png)

# Projeto 2: Hotel Bookings
  O conjunto de dados foi tirado do Kaggle e pode ser acessado [aqui](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand). 
  
  ## Informações
- O conjunto de dados fala sobre as reservas de hotéis de dois tipos ao longo do tempo, sua composição e o status da reserva.
- O Data Set cobre o **período** de **julho de 2015** até **agosto de 2017**.
- **Reservas bem sucedidas**: Marcação e chegada no hotel.
- **Estadia**: Quantidade de dia(s)/tempo que se passou no hotel.
- O **objetivo** desse **projeto** pode ser **dividido** em **duas partes**:
  - **Análise exploratória extensa**: Maior entendimento do comportamento e características dos clientes, para que dessa forma possa favorecer a implementação de melhorias na oferta do serviço e no planejamento da administração. Familiarização com os dados para futura implementação de um modelo de machine learning.
    - A análise exploratória foi divida em 3 partes. Os respectivos **comentários** pertinentes estão presentes nos **respectivos notebooks**. 
  - **Criação e comparação de Modelos de Machine Learning**: Criação de um modelo de machine learning capaz de prever o cancelamento ou não de reservas de hotel. Informação crucial para a administração dos hotéis, facilitando o planejamento das estadias e consequente melhoria da oferta dos serviços além de garantir uma maior segurança no planejamento financeiro.
  - Todas as **considerações, insights e comentários do processo de Machine Learning** pode ser encontrado no **notebook específico** (em markdown) chamado **Hotel Bookings - 0. Parte IV ML- MACHINELEARNING_PROCESS**. Os notebooks posteriores (relacionados ao processo de machine learning) são compostos apenas de código.
  - **TL;DR**:
    - Utilizou-se *class_weight* para lidar com o desbalanceamento de classe.
    - Testou-se a capacidade preditiva com e sem a alteração na cardinalidade da variável *country*. Optou-se por usar o RareLabelEncoder. 
    - Foram utilizados como modelos base: Logistic Regression, Decision Tree e GaussianNB. Posteriormente, utilizou-se modelos ensemble com decision tree como estimador-base (Random Forest, AdaBoost, GradientBossting e XGBoost) que naturalmente obtiveram performances superiores. 
    - Os modelos base iniciais foram comparados entre si, assim como os modelos ensemble. Gráficos de Precision-Recall, Curva ROC, e gráficos de barra das métricas principais: F1-Score, Precision, Recall e Accuracy foram criados para ilustrar a análise.
    - A métrica F1-Score foi a métrica de avaliação considerada como primordial. 
    - Um estimador final (**VotingClassifier**), considerado o ideal, foi construído com base nos ensembles. 
  



### Variáveis presentes:

- *hotel* - Tipo do hotel (Hotel da Cidade ou Resort).
- *is_canceled* - Status da Reserva, se a mesma foi cancelada ou não. 0 -> Não cancelada, 1 -> Cancelada.
- *lead_time* - Número de dias decorridos entre a data de entrada da reserva no PMS (Property Management System) e a data de chegada no hotel.
- *arrival_date_year* - Ano de chegada no hotel.
- *arrival_date_month* - Mês de chegada no hotel.
- *arrival_date_week_number* - Número da semana de chegada no ano.
- *arrival_date_day_of_month* - Data do dia de chegada no hotel.
- *stays_in_weekend_nights* - Estadias aos finais de semana.
- *stays_in_week_nights* - Estadias nos dias de semana. 
- *adults* - Número de adultos que compõem a reserva. 
- *children* - Número de crianças que compõem a reserva. 
- *babies* - Número de bebês que compõem a reserva.
- *meal* - Tipo de refeição escolhida. As categorias são apresentadas em pacotes de refeição de hospitalidade padrão: Undefined/SC (Sem pacote de refeição), BB (Cama & Café da manhã), HB (Café da manhã e mais uma refeição, geralmente jantar) e Full Board (Café da manhã, almoço e jantar).
- *country* - País de origem. As categorias são representadas no formato da ISO 3155–3:2013.
- *market_segment* - Designação do segmento de mercado. Nas categorias, o termo TA significa Agentes de Viagens e o termo TO Operadores de Viagens. 
- *distribution_channel* - Canal de distribuição das reservas. O termo TA significa Agentes de Viagens e o termo TO Operadores de Viagens.
- *is_repeated_guest* - Indica se o hóspede é repedito ou não. 0 -> Não repetido, 1 -> repetido.
- *previous_cancellations* - Número de cancelamentos prévios feitos pelo cliente anteriormente à reserva atual. 
- *previous_bookings_not_canceled* - Número de não cancelamentos prévios feitos pelo cliente anteriormente à reserva atual. 
- *reserved_room_type* - Código do tipo do quarto reservado. É apresentado em forma de código por questões de anonimato. 
- *assigned_room_type* - Código do tipo do quarto que de fato foi atribuído à reserva. Está sujeito a ser diferente do quarto que foi reservado devido a problemas na operacionalização (por exemplo, overbooking) ou até mesmo por solicitação do cliente. É apresentado em código também por motivos de anonimato.   
- *booking_changes* - Número de alterações feitas na reserva desde o momento em que a mesma entrou no PMS até ao momento do check-in ou cancelamento.
- *deposit_type* - Tipo do depósito que o cliente fez (caso tenha feito) para garantir a reserva. No Deposit (Sem depósito), Refundable (Depósito foi feito com um valor inferior ao custo total) e Non Refund (Depósito foi feito com o valor total da estadia). 
- *agent* - ID da agência de viagem que fez a reserva. ID foi apresentado por razões de anonimato.
- *company* - ID da companhia ou entidade que fez a reserva ou se responsabilizou por paga-la. ID foi apresentado por razões de anonimato. 
- *days_in_waiting_list* - Número de dias que a reserva ficou na lista de espera antes de ser confirmada ao cliente.
- *customer_type* - Tipo de reserva, assumindo as seguintes categorias: Transient (quando a reserva não faz parte de um grupo ou contrato), Contract (quando a reserva tem algum tipo de contrato atrelado), Group (quando a reserva é associada a um grupo de pessoas) e Transient-party (quando a reserva é Transient mas é associada com pelo menos outra reseva Transiet).
- *adr* - Taxa diária média. É definida pela divisão da soma de todas as transações de hospedagem pelo número total de noites de hospedagem.
- *required_car_parking_spaces* - Número de vagas de estacionamento exigidas pelo cliente.
- *total_of_special_requests* - Número de pedidos especiais feitos pelo cliente (por exemplo, cama de solteiro ou andar alto).
- *reservation_status* - Último status da reserva. Canceled (Reserva cancelada pelo cliente), Check-Out (O cliente fez check-in mas já partiu) e No-Show (O cliente não fez check-in e informou o hotel sobre o motivo).
- *reservation_status_date* - Data em que o último status da reserva foi estabelecido.

---
#### 💡 Sinta-se livre para contribuir: correções, sugestões, novos insights, muito bem vindos! Valeu! :smiley:
