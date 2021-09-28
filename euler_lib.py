"""--------------------------------DECORATORI--------------------------------"""

class Memoize:
    def __init__(self, f):
        self.f = f
        self.memo = {}
    def __call__(self, *args):
        if not args in self.memo:
            self.memo[args] = self.f(*args)
        #Warning: You may wish to do a deepcopy here if returning objects
        return self.memo[args]


"""----------------------------FUNZIONI ESSENZIALI---------------------------"""

def all_perms(elements):
    """Tutte le possibili permutazioni di una lista"""
    if len(elements) <=1:
        yield elements
    else:
        for perm in all_perms(elements[1:]):
            for i in range(len(elements)):
                # nb elements[0:1] works in both string and list contexts
                yield perm[:i] + elements[0:1] + perm[i:]


def bisezione(f,a,b,toll=10**-5):
    """Trova uno zero della funzione f tra i punti a e b, dove la f assume segno
    discorde. Il parametro opzionale toll indica la precisione con cui si vuole
    calcolare il valore dello zero"""
    m = (a+b)/2
    f_m = f(m)
    while abs(f_m) > toll:
        if f(a)*f_m < 0:
            b = m
        elif f(b)*f_m < 0:
            a = m
        elif f_m == 0:
            print("Trovata solzione esatta")
            return m
        else:
            print("Metodo fallito")
            return None
        m = (a+b)/2
        f_m = f(m)
    return m


def counter(l,b):
    """Incrementa la lista l intesa come numero (sequenza di cifre) contando
    in base b"""
    if l == [b-1]*len(l): return
    i = 1
    while(True):
        if l[-i] < b-1:
            l[-i] += 1
            break
        else:
            l[-i] = 0
            i += 1

def combinations(iterable, r):
    """combinations('ABCD', 2) --> AB AC AD BC BD CD
       combinations(range(4), 3) --> 012 013 023 123"""
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = list(range(r))
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i+1, r):
            indices[j] = indices[j-1] + 1
        yield tuple(pool[i] for i in indices)


def divisori(n):
    """Fornisce i divisori di n"""
    div=set()
    for i in range(1,int(n**0.5+1)):
        if n%i==0:
            div.add(int(n/i))
            div.add(i)
    return sorted(div)


def fact(x):
    """Funzione fattoriale"""
    if x == 0 or x==1: return 1
    ret = 1
    for i in range(x):
        ret *= x-i
    return ret


def first_n_digits(num, n):
    """Prime n cifre del numero num"""
    return num // 10 ** (int(math.log(num, 10)) - n + 1)


def freq(l):
    """Restituisce una lista con le sole frequenze con cui compaiono gli
    elementi nella lista/tupla l. Ad es. se l = (1,2,2,5,3,4,1,1), allora
    freq di l sarà: [3,2,1,1,1]"""
    d = {}
    for i in l:
        if i in d:
            d[i] += 1
        else:
            d[i] = 1
    return list(d.values())

def get_digit(number, n):
    """Restituisce la n-sima cifra del numero number. La 0-esima cifra sono le
    unità, la 1-esima le decine, la 2-esima le centinaia ecc..."""
    return number // 10**n % 10

def is_square(apositiveint):
    """Verifica che un intero positivo sia un quadrato"""
    x = apositiveint // 2
    seen = set([x])
    while x * x != apositiveint:
        x = (x + (apositiveint // x)) // 2
        if x in seen: return False
        seen.add(x)
    return True


def last_n_digits(num, n):
    """Ultime n cifre del numero num"""
    return num%(10**n)


def mcd(a, b):
    """Massimo comune denominatore tra a e b"""
    while(b != 0):
        a,b = b,a%b
    return a


def mobius(n,primes):
    """Restituisce la funzione di Mobius dei primi n numeri. Viene fornita la
    lista primes che deve contenere tutti i primi almeno fino a n, questa cosa
    per velocità non viene verificata, è a cura di chi utilizza la
    funzione!!!"""
    m = [0]*(n+1)
    for p in primes:
        for i in range(p, n+1, p):
            m[i] += 1
    for p in primes:
        p_2 = p**2
        for i in range(p_2, n+1, p_2):
            m[i] = 0
    for i in range(n+1):
        if m[i] == 0:
            continue
        elif m[i]%2 == 0:
            m[i] = 1
        else:
            m[i] = -1
    m[1] = 1
    return m


def multinomial(n, k):
    """Coefficiente multinomiale: dati a,b,...,z elementi della lista k
    restituisce n!/(a! b! ... z!)"""
    ret = fact(n)
    for i in k: ret //= fact(i)
    km = n - sum(k)
    return ret//fact(km)


def newton_dd(l,asc):
    """Newton Divided Differences:
    Restituisce il polinomio di grado n-1 passante per gli n punti contenuti
    nella lista l, calcolato nell'ascissa asc"""
    x = [a[0] for a in l]
    y = [a[1] for a in l]
    n = len(l)
    coeff = [y[0]]
    b = y
    for i in range(1,n):
        b_new = []
        for j in range(1,n-i+1):
            b_new.append((b[j]-b[j-1])/(x[j+i-1]-x[j-1]))
        coeff.append(b_new[0])
        b = copy.deepcopy(b_new)
    ret = 0
    for i in range(len(coeff)):
        t = coeff[i]
        for j in range(i):
            t *= (asc-x[j])
        ret += t
    return ret


def palindromo(s):
    """Dice se una stringa/lista è palindroma"""
    if s == s[::-1]:
        return True
    else:
        return False


def primi(n):
    """Restituisce una lista con tutti i numeri primi fino a n compreso col
    metodo del crivello di Eratostene"""
    numVec = []
    for x in range(n-1):
        numVec.append(x+2)
    for num in numVec[:(n//2-1)]:
        if numVec[num-2] != 0:
            numVec[slice(2*num-2, n-1, num)] = [0]*(n//num-1)
    numVec = [x for x in numVec if x!=0]
    return numVec


def primo(num):
    """Dice se un numero è primo"""
    if num <= 1: return False
    if num == 2: return True
    for i in range(2,int(num**(1/2)+1)):
        if num%i == 0:
            return False
    return True


def radicale(n):
    """Restituisce il radicale di n"""
    r = 1
    for p in primi(n+1):
        if p>n:
            break
        if n%p==0:
            r *= p
            n = n//p
    return r


def simpson_13(f,a,b,n=1):
    """Algoritmo di integrazione Simpson 1/3, per calcolare l'integrale della
    funzione f tra gli estremi a e b. L'intervallo è suddiviso in n
    sotto-intervalli. Aumentando n aumenta la precisione."""
    step = (b-a)/n
    h = step/2
    ret = 0
    for i in range(n):
        a_1 = a + i*step
        b_1 = a_1 + step
        ret += h/3*(f(a_1)+4*f(a_1+h)+f(b_1))
    return ret


"""------------------------------ALTRE FUNZIONI------------------------------"""

def collatz(n):
    """Fornisce l'elemento successivo nella sequenza di Collatz"""
    if n%2==0: return n/2
    else: return 3*n+1


def counter_desc(l,b):
    """Incrementa la lista l intesa come numero (sequenza di cifre) contando
    in base b e accettando solo numeri in cui ogni cifra è maggiore o uguale
    della cifra alla sua destra"""
    if l == [b-1]*len(l): return
    i = 1
    ll = len(l)
    while(True):
        if i==ll: n_max = b-1
        else: n_max = l[-i-1]
        if l[-i] < n_max:
            l[-i] += 1
            break
        else:
            l[-i] = 0
            i += 1


def kruskal(m):
    """Algoritmo di kruskal per la ricerca dell'MST di un grafo, fornito in
    tramite la sua matrice di adiacenza, usa la funzione ring_finder per cercare
    anelli nel grafo e di min_nonzero_idx per trovare gli inidici dei rami con
    costo minimo"""
    n = m.shape[0]
    m_ret = np.zeros([n,n], dtype=int)
    while np.count_nonzero(m_ret) != 2*(n-1):
        i_min, j_min = min_nonzero_idx(m)
        n_min = m[i_min, j_min]
        m[i_min, j_min], m[j_min, i_min] = 0, 0
        m_ret[i_min, j_min], m_ret[j_min, i_min] = n_min, n_min
        if ring_finder(m_ret, [i_min], []):
            m_ret[i_min, j_min], m_ret[j_min, i_min] = 0, 0
    return m_ret


def miller_rabin(n,k):
    """Implementation uses the Miller-Rabin Primality Test
    The optimal number of rounds for this test is 40
    See http://stackoverflow.com/questions/6325576/how-many-iterations-of-rabin-miller-should-i-use-for-cryptographic-safe-primes
    for justification"""
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    r, s = 0, n - 1
    while s % 2 == 0:
        r += 1
        s //= 2
    for _ in range(k):
        a = random.randrange(2, n - 1)
        x = pow(a, s, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def min_nonzero_idx(m):
    """Restituisce la posizione del valore più piccolo diverso da zero presente
    nella matrice"""
    n, ret, minval = m.shape[0], [0,0], m.max()
    for i in range(n):
        for j in range(n):
            val = m[i, j]
            if val < minval and val != 0:
                minval, ret = val, (i, j)
    return ret


def n_modi(t):
    """Restituisce il numero di modi in cui 1/n è scrivibile come 1/x + 1/y
    con n, x e y interi. L'argomento t è la tupla contenete la frequenza con cui
    ogni fattore primo compare nella scomposizione di n (fattori primi)"""
    ret = 1
    for i in t:
        ret += i*(2*ret-1)
    return ret


def periodic_sqrt(n):
    """Ritorna una lista con lo sviluppo in frazioni continue della radice di
    n: il primo elemento è il termine di partenza ed è seguito dal periodo, il
    secondo elemento è la lunghezza del periodo"""
    if n**0.5 == int(n**0.5): return [int(n**0.5)], 0
    res = [int(n**0.5)]
    a, b = 1, res[0]
    while True:
        num = a*(b+n**0.5)
        den = n - b**2
        if mcd(a,den)!=0:
            den_com = mcd(a,den)
            a /= den_com
            den /= den_com
        for j in range(res[0],-1,-1):
            if (b+j)%den==0:
                res.append(int((b+j)/den))
                b = j
                break
        if res[-1]==2*res[0]:
            return res, len(res)-1
        a = den


def piramide_pascal(n):
    """Restituisce una lista con i primi n livelli della piramide di Pascal,
    ogni livello è una matrice triangolare"""
    n+=1
    pir = [np.array([[1]])]
    for n_0 in range(2,n):
        tri_old = pir[n_0-2]
        tri = np.zeros([n_0,n_0],dtype=int)
        tri[0][0], tri[0][n_0-1], tri[n_0-1][0] = 1, 1, 1
        for i in range(1,n_0-1): # inizializzo i lati
            a = tri_old[0][i-1] + tri_old[0][i]
            tri[0][i] = tri[i][0] = tri[i][n_0-i-1] = a
        for i in range(1,n_0-2):
            for j in range(1,n_0-2):
                tri[i][j] = tri_old[i][j] + tri_old[i-1][j] + tri_old[i][j-1]
        pir.append(tri)
    return pir


def pff(n):
    """PRIME FACTORS OF FACTORIAL
    Restituisce una lista in cui all'i-esima posizione si trovala potenza
    a cui l'i-esimo numero primo è elevato nella scomposizione in fattori
    primi del fattoriale di n"""
    global primes
    #primes = primi(n)
    ret = []
    for p in primes:
        a = 0
        if p > n:
            break
        for i in range(1, int(log(n,p)) + 1):
            a += int(n/p**i)
        ret.append(a)
    return ret


def pfb(n1, n2):
    """PRIME FACTORS OF BINOMIAL
    Restituisce una lista in cui all'i-esima posizione si trovala potenza
    a cui l'i-esimo numero primo è elevato nella scomposizione in fattori
    primi del coefficiente binomiale di (n1 n2)"""
    global primes
    #primes = primi(n)
    n3 = n1 - n2
    factors_1, factors_2, factors_3 = pff(n1), pff(n2), pff(n3)
    for i in range(len(factors_2)):
        factors_1[i] -= factors_2[i]
        if i < len(factors_3):
            factors_1[i] -= factors_3[i]
    return factors_1


def ring_finder(arr, vertices, edges):
    """Algoritmo ricorsivo per la ricerca di anelli all'interno di un grafo,
    il primo argomento è la matrice di adiacenza del grafo. La lista vertices
    deve venire fornita in input contenente un solo numero, che indica il nodo
    da cui si parte per cercare l'anello (la colonna/riga del nodo nella matrice
    di adiacenza)"""
    n, j = arr.shape[0], vertices[-1]
    for i in range(n):
        edge = arr[i,j]
        if edge == 0:
            continue
        elif not i in vertices:
            if ring_finder(arr, vertices + [i], edges + [arr[i, j]]):
                return True
        elif i == vertices[0] and len(vertices) > 2:
            #print("*** ANELLO TROVATO ***")
            return True
    return False


def triangolo_pascal(n):
    """Restituisce una lista con le prime n righe del triangolo di Pascal, ogni
    riga è una a sua volta una lista"""
    tri = [[1]]
    for i in range(1,n):
        tri.append([1])
        for j in range(1,i):
            tri[i].append(tri[i-1][j-1] +tri[i-1][j])
        tri[i].append(1)
    return tri
