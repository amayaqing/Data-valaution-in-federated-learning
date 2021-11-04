# Data-valaution-in-federated-learning

**CityU final year thesis project**

Supervisor: Prof. Wang Cong



Recently, federated learning (FL) emerges as a promising framework to collect the dispersed data and train a collaborative machine learning (ML) model with privacy protection. An incentive scheme plays a crucial role in the FL system as they encourage long-term client joining. However, due to information asymmetry between the central server and local users, a key challenge is to evaluate participants’ contributions in an objective and efficient manner so as to allocate the payoff fairly. Data valuation in ML context is a systematic study on quantifying the usefulness of a specific data point in a prediction model. It provides a potential solution for FL to measure local client’s quality. However, exponential computational complexity and additional communication costs are critical challenges of applying data valuation-based incentive schemes.

In this project, we propose a new round-based data valuation (RDV) approach to serve as a real-time incentive mechanism. It takes advantage of the FL system’s unique model aggregation property to increase the valuation efficiency and provide a fine-grained contribution estimation on a per-round basis. It also offers a guideline for the central server to selectively aggregate the local updates to train a better-performing model. We empirically demonstrate the effectiveness of RDV in identifying high-quality participants, the efficiency in allocating payoff, and its potentials in federation optimization.
