import streamlit as st
import numpy as np
# Import matplotlib dihapus karena menggunakan Plotly agar lebih interaktif di web
import plotly.graph_objects as go
from scipy import optimize
import sympy as sp

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Metode Numerik Companion",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS CUSTOM ---
st.markdown("""
<style>
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feynman-box {
        background-color: #f0f2f6;
        border-left: 5px solid #ff6b6b;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }
    .stButton>button {
        color: white;
        background-color: #1f77b4;
        border-radius: 5px;
        height: 3em;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# --- FUNGSI NUMERIK ---

def parse_expression(expr_str):
    """Mengubah string input user menjadi fungsi python yang bisa dieksekusi"""
    try:
        # Deteksi umum kesalahan pemula
        if '^' in expr_str:
            # Kita tidak otomatis ganti agar user belajar, tapi kita deteksi
            pass 
            
        allowed_names = {
            "sin": np.sin, "cos": np.cos, "tan": np.tan, "exp": np.exp, "log": np.log,
            "sqrt": np.sqrt, "pi": np.pi, "e": np.e, "abs": np.abs
        }
        f = lambda x: eval(expr_str, {"__builtins__": None}, allowed_names)
        
        # Jalankan tes singkat untuk memastikan tidak crash saat plot
        _ = f(np.array([0.0, 1.0]))
        return f
        
    except SyntaxError:
        st.error("❌ **Syntax Error:** Cek kembali penulisan fungsi Anda.")
        return None
    except TypeError as e:
        st.error(f"❌ **TypeError:** {e}. Gunakan `**` untuk pangkat (x**2), bukan `^`.")
        return None
    except Exception as e:
        st.error(f"❌ **Error:** {e}")
        return None

def bisection_method(f, a, b, tol, max_iter):
    """Implementasi Algoritma Bisection"""
    results = []
    if f(a) * f(b) > 0:
        return None, "Akar tidak terdampar dalam interval ini (f(a)*f(b) harus negatif)"
    
    for i in range(max_iter):
        c = (a + b) / 2
        results.append((i+1, a, b, c, f(c)))
        
        if abs(f(c)) < tol or (b - a)/2 < tol:
            return results, "Konvergen"
        
        if f(c) * f(a) < 0:
            b = c
        else:
            a = c
            
    return results, "Maksimum iterasi tercapai"

def newton_raphson_method(f, df, x0, tol, max_iter):
    """Implementasi Algoritma Newton-Raphson"""
    results = []
    x = x0
    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)
        if dfx == 0:
            return None, "Turunan nol (pembagian dengan nol)"
        
        x_new = x - fx / dfx
        results.append((i+1, x, fx, x_new, abs(x_new - x)))
        
        if abs(x_new - x) < tol:
            return results, "Konvergen"
        x = x_new
        
    return results, "Maksimum iterasi tercapai"

# --- SIDEBAR NAVIGASI ---
st.sidebar.title("Menu Materi")
menu = st.sidebar.radio(
    "Navigasi Pembelajaran:",
    ["Beranda", "Analisis Galat", "Akar Persamaan", "Sistem Linear", "Interpolasi", "Integral & PDB"]
)

# --- HALAMAN 1: BERANDA ---
if menu == "Beranda":
    st.markdown('<div class="main-title">Metode Numerik Companion</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("""
    ### Selamat Datang, Insinyur Muda! 👋
    
    Ini adalah teman belajar interaktif untuk mata kuliah **Metode Numerik (ELT60214)**.
    Aplikasi ini dirancang untuk membantu Anda memahami bagaimana komputer menyelesaikan 
    masalah matematika yang "terlalu sulit" diselesaikan dengan tangan.
    """)
    
    st.markdown('<div class="feynman-box">', unsafe_allow_html=True)
    st.markdown("""
    ### 💡 Konsep Feynman: Mengapa Kita Butuh Ini?
    
    Bayangkan Anda ingin menghitung luas area kolam renang yang berbentuk sangat aneh, 
    bukan lingkaran atau kotak biasa. Rumus Luas = panjang x lebar tidak bisa digunakan.
    
    Di sinilah **Metode Numerik** berperan. Alih-alih mencari satu jawaban "tepat" 
    (yang mungkin tidak ada), kita mencari jawaban "cukup dekat" yang bisa diterima 
    dalam dunia teknik.
    
    *Prinsip Utama:* **Approximation (Pendekatan) + Iterasi (Pengulangan) = Solusi.**
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    st.info("Gunakan menu di sebelah kiri untuk memulai eksperimen dengan metode-metode numerik.")

# --- HALAMAN 2: ANALISIS GALAT ---
elif menu == "Analisis Galat":
    st.header("🔍 Analisis Galat (Error Analysis)")
    
    st.markdown("""
    Sebelum kita menghitung apapun, kita harus tahu: **Seberapa salah bolehnya jawaban kita?**
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="concept-title">Konsep Absolute Error</div>', unsafe_allow_html=True)
        st.latex(r"E_{abs} = |x_{true} - x_{approx}|")
        st.caption("Ini adalah selisih murni. Misalnya, beda 1 meter. Tapi, beda 1 meter di lapangan sepak bola itu kecil, beda 1 meter saat operasi bedah itu bencana.")
        
    with col2:
        st.markdown('<div class="concept-title">Konsep Relative Error</div>', unsafe_allow_html=True)
        st.latex(r"E_{rel} = \left| \frac{x_{true} - x_{approx}}{x_{true}} \right| \times 100\%")
        st.caption("Ini memberi konteks skala. Error 1mm pada pemasangan mikrochip besar, tapi kecil pada pembangunan jembatan.")
    
    st.markdown("---")
    
    true_val = st.number_input("Masukkan Nilai Sebenarnya (True Value):", value=100.0)
    approx_val = st.number_input("Masukkan Nilai Pendekatan (Approximate):", value=98.5)
    
    abs_err = abs(true_val - approx_val)
    rel_err = (abs_err / true_val) * 100 if true_val != 0 else 0
    
    st.success(f"Absolute Error: {abs_err:.4f}")
    st.success(f"Relative Error: {rel_err:.4f}%")

# --- HALAMAN 3: AKAR PERSAMAAN ---
elif menu == "Akar Persamaan":
    st.header("🔎 Mencari Akar (Root Finding)")
    
    st.markdown("""
    Masalah klasik: Kapan fungsi ini menyentuh sumbu X (nilai nol)?
    Dalam teknik: Kapan arus nol? Kapan tegangan drop menjadi nol?
    """)
    
    method = st.radio("Pilih Metode:", ["Bisection (Dikotomi)", "Newton-Raphson"])
    
    # Input Fungsi
    func_input = st.text_input(
        "Masukkan fungsi f(x):", 
        value="x**2 - 4",
        help="Tips: Gunakan `**` untuk pangkat. Contoh: x**2 - 4"
    )
    
    # Parse Fungsi
    f = parse_expression(func_input)
    
    # CEK KEGAGALAN: Jika f gagal diproses, berhenti di sini.
    if f is None:
        st.error("❌ **Gagal Memproses Fungsi.** Periksa penulisan Anda.")
        st.caption("Pastikan menggunakan `**` (bintang dua) untuk pangkat, bukan `^`.")
        st.stop()  # INI PENTING: Mencegah error lanjutan 'NoneType'
        
    # Jika sampai sini berarti fungsi valid, kita baru hitung plot dasar
    try:
        x = np.linspace(-10, 10, 400)
        y = f(x)
        
        # Plot Fungsi Awal
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='f(x)'))
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(title="Visualisasi Fungsi", xaxis_title='x', yaxis_title='f(x)')
        st.plotly_chart(fig, use_container_width=True)
    except Exception as plot_err:
        st.error(f"❌ Error saat memplot fungsi: {plot_err}")
        st.stop()

    # LOGIKA BISECTION
    if method == "Bisection (Dikotomi)":
        st.markdown("""
        <div class="feynman-box">
        <b>Analogi: Permainan Panas Dingin.</b><br>
        Kita punya dua titik A dan B. Di A nilainya positif, di B negatif. 
        Artinya di tengah-tengah mereka, pasti ada titik yang nilainya NOL.
        Kita potong tengahnya, lalu buang bagian yang tidak relevan.
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        a = col1.number_input("Batas Bawah (a):", value=-3.0)
        b = col2.number_input("Batas Atas (b):", value=0.0)
        tol = col1.number_input("Toleransi:", value=0.001)
        max_iter = col2.number_input("Max Iterasi:", value=10, min_value=1)
        
        if st.button("Hitung Akar (Bisection)"):
            results, msg = bisection_method(f, a, b, tol, max_iter)
            if results is None:
                st.error(f"Gagal: {msg}")
            else:
                st.success(f"Status: {msg}")
                st.write("Tabel Iterasi:")
                st.dataframe(results, columns=["Iter", "a", "b", "c (Akar)", "f(c)"])
                
                # Animasi Sederhana pada Plot
                last_c = results[-1][2]
                fig.add_trace(go.Scatter(x=[last_c], y=[0], mode='markers', marker=dict(size=15, color='red'), name='Akar Ditemukan'))
                st.plotly_chart(fig, use_container_width=True)

    # LOGIKA NEWTON-RAPHSON
    elif method == "Newton-Raphson":
        st.markdown("""
        <div class="feynman-box">
        <b>Analogi: Meluncur di Bukit.</b><br>
        Kita mulai dari titik tebakan. Kita tarik garis singgung (turunan). 
        Garis singgung itu akan memotong sumbu X di titik baru. Titik baru ini 
        pasti lebih dekat ke akar yang asli daripada titik awal.
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        x0 = col1.number_input("Tebakan Awal (x0):", value=3.0)
        tol = col2.number_input("Toleransi:", value=0.001)
        
        if st.button("Hitung Akar (Newton-Raphson)"):
            # Numerical derivative untuk efisiensi user
            h = 1e-5
            df = lambda x: (f(x+h) - f(x-h)) / (2*h)
            
            results, msg = newton_raphson_method(f, df, x0, tol, max_iter=10)
            
            if results is None:
                st.error(f"Gagal: {msg}")
            else:
                st.success(f"Status: {msg}")
                st.write("Tabel Iterasi:")
                st.dataframe(results, columns=["Iter", "x_old", "f(x)", "x_new", "Delta"])
                
                last_x = results[-1][2]
                fig.add_trace(go.Scatter(x=[last_x], y=[0], mode='markers', marker=dict(size=15, color='green'), name='Akar Ditemukan'))
                st.plotly_chart(fig, use_container_width=True)

# --- HALAMAN 4: SISTEM LINEAR ---
elif menu == "Sistem Linear":
    st.header("🧮 Sistem Persamaan Linear")
    
    st.markdown("""
    Memecahkan masalah seperti: "Berapa arus di setiap cabang rangkaian?"
    Ini adalah bentuk matrix $Ax = b$.
    """)
    
    st.info("Saat ini kita simulasi untuk sistem 3x3 (3 Persamaan, 3 Variabel).")
    
    col1, col2, col3 = st.columns(3)
    a11 = col1.number_input("a11", value=3.0)
    a12 = col1.number_input("a12", value=2.0)
    a13 = col1.number_input("a13", value=-1.0)
    b1 = col1.number_input("b1", value=1.0)
    
    a21 = col2.number_input("a21", value=2.0)
    a22 = col2.number_input("a22", value=-2.0)
    a23 = col2.number_input("a23", value=4.0)
    b2 = col2.number_input("b2", value=-2.0)
    
    a31 = col3.number_input("a31", value=-1.0)
    a32 = col3.number_input("a32", value=0.5)
    a33 = col3.number_input("a33", value=-1.0)
    b3 = col3.number_input("b3", value=0.0)
    
    if st.button("Selesaikan dengan Gauss (via NumPy)"):
        A = np.array([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])
        b = np.array([b1, b2, b3])
        
        try:
            sol = np.linalg.solve(A, b)
            st.success(f"Solusi x = {sol}")
            
            # Menampilkan Sistem
            st.latex(fr"""
            \begin{{bmatrix}} {a11} & {a12} & {a13} \\ {a21} & {a22} & {a23} \\ {a31} & {a32} & {a33} \end{{bmatrix}}
            \begin{{bmatrix}} x_1 \\ x_2 \\ x_3 \end{{bmatrix}} =
            \begin{{bmatrix}} {b1} \\ {b2} \\ {b3} \end{{bmatrix}}
            """)
            
            st.write("NumPy menggunakan metode LAPACK (sebagaimana Gauss) yang sangat optimal.")
        except np.linalg.LinAlgError:
            st.error("Singular Matrix! Tidak ada solusi unik (determinan 0).")

# --- HALAMAN 5: INTERPOLASI ---
elif menu == "Interpolasi":
    st.header("📈 Interpolasi & Regresi")
    
    st.markdown("""
    <div class="feynman-box">
    <b>Analogi: Melubangi Kertas.</b><br>
    - <b>Interpolasi:</b> Saya memberi Anda 5 titik lubang di kertas, Anda gambar garis yang <b>harus</b> tembus tepat di lubang tersebut.
    - <b>Regresi:</b> Saya berikan 50 titik (yang berantakan karena error), Anda gambar garis yang paling "adil" di tengah-tengah kerumunan titik tersebut (tidak harus tembus semua titik).
    </div>
    """, unsafe_allow_html=True)
    
    x_vals = st.text_input("Masukkan data X (pisahkan koma):", value="1, 2, 3, 4")
    y_vals = st.text_input("Masukkan data Y (pisahkan koma):", value="1, 4, 9, 16") # Ini sebenarnya kuadrat sempurna
    
    try:
        x_data = np.array([float(i) for i in x_vals.split(',')])
        y_data = np.array([float(i) for i in y_vals.split(',')])
        
        # Lagrange Interpolation
        poly = np.poly1d(np.polyfit(x_data, y_data, len(x_data)-1))
        
        # Regression (Linear)
        poly_reg = np.poly1d(np.polyfit(x_data, y_data, 1))
        
        x_range = np.linspace(min(x_data)-1, max(x_data)+1, 100)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='markers', name='Data Asli'))
        fig.add_trace(go.Scatter(x=x_range, y=poly(x_range), mode='lines', name='Interpolasi (Poly Fit)', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=x_range, y=poly_reg(x_range), mode='lines', name='Regresi Linear', line=dict(color='green', dash='dot')))
        
        st.plotly_chart(fig, use_container_width=True)
        st.info("Perhatikan garis Merah (Interpolasi) tembus semua titik, sedangkan Hijau (Regresi) mengambil tren.")
        
    except ValueError:
        st.warning("Mohon periksa format input angka.")

# --- HALAMAN 6: INTEGRAL & PDB ---
elif menu == "Integral & PDB":
    st.header("⚙️ Integral & Persamaan Differensial (ODE)")
    
    tab1, tab2 = st.tabs(["Integral Numerik", "Simulasi Rangkaian RC (PDB)"])
    
    with tab1:
        st.markdown("""
        Konsep Feynman: "Jumlahan Kecil-Kecil". 
        Kita tidak tahu rumus area bawah kurva, tapi kita tahu lebar-nya kecil (dx) dan tingginya (f(x)).
        Kalikan saja, lalu jumlahkan semuanya. Itu Integral.
        """)
        
        func_input = st.text_input("Fungsi f(x):", value="np.sin(x) + 1")
        f = parse_expression(func_input)
        a = st.number_input("Batas Bawah (a):", value=0.0)
        b = st.number_input("Batas Atas (b):", value=3.14) # Pi
        n = st.number_input("Jumlah Segmen (n):", value=10, min_value=4)
        
        x = np.linspace(a, b, n+1)
        y = f(x)
        
        # Trapezoidal Rule
        dx = (b - a) / n
        integral = 0.5 * dx * (y[0] + 2*sum(y[1:-1]) + y[-1])
        
        # Plotting visual trapezoids using Plotly
        fig = go.Figure()
        x_continuous = np.linspace(a, b, 200)
        fig.add_trace(go.Scatter(x=x_continuous, y=f(x_continuous), mode='lines', name='f(x)'))
        
        for i in range(n):
            fig.add_trace(go.Scatter(
                x=[x[i], x[i], x[i+1], x[i+1]], 
                y=[0, y[i], y[i+1], 0], 
                mode='lines', fill='toself', fillcolor='rgba(0,100,80,0.2)', showlegend=False
            ))
            
        st.plotly_chart(fig, use_container_width=True)
        st.success(f"Hasil Integral Numerik (Trapezoidal): {integral:.5f}")
        
    with tab2:
        st.markdown("""
        Studi Kasus UAS: **Respon Rangkaian RC**.
        
        Persamaan: $dV_c/dt = (V_{in} - V_c) / RC$         Kita gunakan metode **Euler** untuk mensimulasikan langkah demi langkah.
        """)
        
        R = st.slider("Resistansi R (kOhm):", 1.0, 20.0, 10.0) * 1000
        C = st.slider("Kapasitansi C (uF):", 10.0, 500.0, 100.0) * 1e-6
        Vin = 5.0 # Step input
        tau = R * C
        
        st.info(f"Time Constant (tau) = R x C = {tau:.2f} detik")
        st.write("Secara teori, capacitor akan penuh sekitar 5 tau.")
        
        # Euler Method Simulation
        h = 0.01 # time step
        t_max = 5 * tau
        steps = int(t_max / h)
        
        t_vals = [0]
        vc_vals = [0] # Awal Vc = 0
        
        vc = 0
        for i in range(steps):
            dvc_dt = (Vin - vc) / tau
            vc = vc + dvc_dt * h # Euler update
            t_vals.append(t_vals[-1] + h)
            vc_vals.append(vc)
            
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t_vals, y=vc_vals, mode='lines', name='Vc(t) Simulasi'))
        fig.add_hline(y=Vin, line_dash="dash", annotation_text="Vin (5V)")
        fig.update_layout(xaxis_title="Waktu (s)", yaxis_title="Tegangan Kapasitor (V)")
        st.plotly_chart(fig, use_container_width=True)

# --- FOOTER ---
st.sidebar.markdown("---")
st.sidebar.write("Dibuat untuk:")
st.sidebar.write("S1 Teknik Elektro - ELT60214")
st.sidebar.write("© 2025 Metode Numerik")




