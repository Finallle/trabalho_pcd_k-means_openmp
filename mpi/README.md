# Executar e compilar K-means 1-d com MPI

## Apenas em um host
```bash
make single
```

ou com os comandos `mpicc` e `mpirun`.

```bash
mpicc -02 -std=c99 kmeans_1d_mpi.c -o kmeans_1d_mpi -lm
mpirun -np 2 ./kmeans_1d_mpi dados_big.csv centroides_k64.csv 50 0.000001 assign.csv cent
```

## Em vários hosts

### Configuração
É necessário ter uma conexão SSH direta entre os hosts e configurar o arquivo `hosts`, com o endereço de IP local e o número de cores de cada dispositivo (através de `slots`). 
```
192.168.0.25 slots=8
192.168.0.28 slots=2
```
Você pode verificar o IP com o comando `ifconfig`.

Além disso, é necessário iniciar o serviço do SSH em ambos os hosts.

Com OpenSSH:
```bash
sudo systemctl start sshd
```

Crie uma chave pública e privada entre os dispositivos, para que evite a autenticação com usuário e senha. No exemplo abaixo, é usado o método RSA para criptografia.

```bash
ssh-keygen -t rsa -b 4096
cd $HOME
cd .ssh
cat id_rsa.pub > authorized_keys
```

No <i>localhost</i> atual, envie as chaves para o outro host com o comando `scp`.
```
export IP=IP_DO_OUTRO_HOST
cd ~/.ssh
scp usuario@$IP:$HOME/.ssh/id_rsa.pub chave-vizinha.txt
cat chave-vizinha.txt >> authorized_keys
scp authorized_keys usuario@$IP:$HOME/.ssh/authorized_keys
```

## Execução

Volte para a pasta `mpi` e execute o programa.
```bash
make hosts
```

ou com os comandos `mpicc` e `mpirun`.

```bash
mpicc -02 -std=c99 kmeans_1d_mpi.c -o kmeans_1d_mpi -lm
mpirun -np 8 --hostfile hosts ./kmeans_1d_mpi dados_big.csv centroides_k64.csv 50 0.000001 assign.csv cent
```

# Avisos
* O parâmetro `-np` definirá o número de processos a serem executados.
