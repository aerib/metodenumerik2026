import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy import optimize
import sympy as sp
import pandas as pd

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
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 20px;
    }
    .concept-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 10px 0;
    }
    .formula-box {
        background-color: #e8eaf6;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# --- FUNGSI NUMERIK ---
def parse_expression(expr_str):
    """Mengubah string input user menjadi fungsi python yang bisa dieksekusi"""
    try:
        # Deteksi kesalahan umum
        if '^' in expr_str:
            st.error("❌ Gunakan `**` untuk pangkat, bukan `^`. Contoh: x**2 bukan x^2")
            return None
        
        # Replace 'x' dengan array placeholder untuk evaluasi
        allowed_names = {
            "sin": np.sin,
            "cos": np.cos,
            "tan": np.tan,
            "exp": np.exp,
            "log": np.log,
            "sqrt": np.sqrt,
            "pi": np.pi,
            "e": np.e,
            "abs": np.abs,
        }
        
        # Buat fungsi lambda yang benar
        def f(x):
            # Buat namespace baru untuk setiap evaluasi
            namespace = allowed_names.copy()
            namespace['x'] = x
            return eval(expr_str, {"__builtins__": None}, namespace)
        
        # Test dengan array untuk memastikan fungsi bekerja
        test_array = np.array([0.0, 1.0, 2.0])
        test_result = f(test_array)
        
        # Pastikan hasilnya adalah array
        if not isinstance(test_result, np.ndarray):
            test_result = np.array(test_result)
        
        return f
        
    except SyntaxError as e:
        st.error(f"❌ **Syntax Error:** Periksa penulisan fungsi Anda. Detail: {e}")
        return None
    except NameError as e:
        st.error(f"❌ **Name Error:** Variabel atau fungsi tidak dikenal. Gunakan 'x' sebagai variabel. Detail: {e}")
        return None
    except TypeError as e:
        st.error(f"❌ **Type Error:** {e}. Gunakan `**` untuk pangkat (x**2), bukan `^`.")
        return None
    except Exception as e:
        st.error(f"❌ **Error:** {e}")
        return None

def bisection_method(f, a, b, tol, max_iter):
    """Implementasi Algoritma Bisection"""
    results = []
    
    try:
        fa = float(f(a))
        fb = float(f(b))
    except Exception as e:
        return None, f"Error mengevaluasi fungsi: {e}"
    
    if fa * fb > 0:
        return None, "⚠️ Akar tidak terdapat dalam interval ini (f(a) dan f(b) harus berlawanan tanda)"
    
    for i in range(max_iter):
        c = (a + b) / 2
        fc = float(f(c))
        results.append((i+1, a, b, c, fc))
        
        if abs(fc) < tol or (b - a)/2 < tol:
            return results, "✅ Konvergen"
        
        if fc * fa < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
    
    return results, "⚠️ Maksimum iterasi tercapai"

def newton_raphson_method(f, df, x0, tol, max_iter):
    """Implementasi Algoritma Newton-Raphson"""
    results = []
    x = x0
    
    for i in range(max_iter):
        try:
            fx = float(f(x))
            dfx = float(df(x))
        except Exception as e:
            return None, f"Error mengevaluasi fungsi: {e}"
        
        if abs(dfx) < 1e-10:
            return None, "⚠️ Turunan mendekati nol (pembagian dengan nol)"
        
        x_new = x - fx / dfx
        results.append((i+1, x, fx, x_new, abs(x_new - x)))
        
        if abs(x_new - x) < tol:
            return results, "✅ Konvergen"
        
        x = x_new
    
    return results, "⚠️ Maksimum iterasi tercapai"

def secant_method(f, x0, x1, tol, max_iter):
    """Implementasi Algoritma Secant"""
    results = []
    
    for i in range(max_iter):
        try:
            f0 = float(f(x0))
            f1 = float(f(x1))
        except Exception as e:
            return None, f"Error mengevaluasi fungsi: {e}"
        
        if abs(f1 - f0) < 1e-10:
            return None, "⚠️ Pembagian dengan nol"
        
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        f2 = float(f(x2))
        results.append((i+1, x0, x1, x2, f2))
        
        if abs(x2 - x1) < tol:
            return results, "✅ Konvergen"
        
        x0, x1 = x1, x2
    
    return results, "⚠️ Maksimum iterasi tercapai"

# --- SIDEBAR NAVIGASI ---
st.sidebar.title("📚 Menu Materi")
menu = st.sidebar.radio(
    "Navigasi Pembelajaran:",
    ["🏠 Beranda", "🔍 Analisis Galat", "🎯 Akar Persamaan", "🧮 Sistem Linear", "📈 Interpolasi", "⚙️ Integral & PDB"]
)

# --- HALAMAN 1: BERANDA ---
if menu == "🏠 Beranda":
    st.markdown('<div class="main-header">Metode Numerik Companion</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("""
    ### Selamat Datang, Insinyur Muda! 👋
    
    Ini adalah teman belajar interaktif untuk mata kuliah **Metode Numerik (ELT60214)**. 
    Aplikasi ini dirancang untuk membantu Anda memahami bagaimana komputer menyelesaikan 
    masalah matematika yang "terlalu sulit" diselesaikan dengan tangan.
    """)
    
    st.markdown('<div class="concept-box">', unsafe_allow_html=True)
    st.markdown("""
    ### 💡 Konsep Feynman: Mengapa Kita Butuh Ini?
    
    Bayangkan Anda ingin menghitung luas area kolam renang yang berbentuk sangat aneh, 
    bukan lingkaran atau kotak biasa. Rumus Luas = panjang × lebar tidak bisa digunakan. 
    
    Di sinilah **Metode Numerik** berperan. Alih-alih mencari satu jawaban "tepat" 
    (yang mungkin tidak ada), kita mencari jawaban "cukup dekat" yang bisa diterima 
    dalam dunia teknik.
    
    **Prinsip Utama:** 
    > Approximation (Pendekatan) + Iterasi (Pengulangan) = Solusi
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("🔍 **Analisis Galat**\n\nBelajar mengukur seberapa akurat hasil perhitungan numerik")
    
    with col2:
        st.success("🎯 **Akar Persamaan**\n\nMencari nilai x dimana f(x) = 0")
    
    with col3:
        st.warning("🧮 **Sistem Linear**\n\nMenyelesaikan sistem persamaan linear")
    
    st.markdown("---")
    st.info("💡 **Tip:** Gunakan menu di sidebar untuk memulai eksperimen dengan metode-metode numerik!")

# --- HALAMAN 2: ANALISIS GALAT ---
elif menu == "🔍 Analisis Galat":
    st.header("🔍 Analisis Galat (Error Analysis)")
    
    st.markdown("""
    Sebelum kita menghitung apapun, kita harus tahu: **Seberapa salah bolehnya jawaban kita?**
    
    Dalam komputasi numerik, kita tidak bisa mendapatkan jawaban yang 100% tepat. 
    Yang penting adalah memahami dan mengontrol tingkat kesalahan.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="formula-box"><h4>Absolute Error</h4></div>', unsafe_allow_html=True)
        st.latex(r"E_{abs} = |x_{true} - x_{approx}|")
        st.caption("""
        **Absolute Error** adalah selisih murni antara nilai sebenarnya dan nilai pendekatan.
        
        **Contoh:** Jika nilai sebenarnya 100 dan hasil perhitungan 98, maka absolute error = 2.
        
        **Kapan penting?** Ketika skala pengukuran konsisten (misal: semua dalam meter).
        """)
    
    with col2:
        st.markdown('<div class="formula-box"><h4>Relative Error</h4></div>', unsafe_allow_html=True)
        st.latex(r"E_{rel} = \left| \frac{x_{true} - x_{approx}}{x_{true}} \right| \times 100\%")
        st.caption("""
        **Relative Error** memberi konteks skala pada kesalahan.
        
        **Contoh:** Error 1mm pada mikrochip (besar!), tapi kecil pada jembatan.
        
        **Kapan penting?** Ketika membandingkan error pada skala yang berbeda.
        """)
    
    st.markdown("---")
    st.subheader("🧪 Kalkulator Error")
    
    col1, col2 = st.columns(2)
    
    with col1:
        true_val = st.number_input("Nilai Sebenarnya (True Value):", value=100.0, format="%.6f")
    
    with col2:
        approx_val = st.number_input("Nilai Pendekatan (Approximate):", value=98.5, format="%.6f")
    
    if true_val != 0:
        abs_err = abs(true_val - approx_val)
        rel_err = (abs_err / abs(true_val)) * 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Absolute Error", f"{abs_err:.6f}")
        
        with col2:
            st.metric("Relative Error", f"{rel_err:.6f}%")
        
        # Interpretasi
        st.markdown("---")
        st.subheader("📊 Interpretasi")
        
        if rel_err < 0.1:
            st.success("✅ **Sangat Baik!** Relative error < 0.1%")
        elif rel_err < 1:
            st.info("✓ **Baik.** Relative error < 1%")
        elif rel_err < 5:
            st.warning("⚠️ **Cukup.** Relative error < 5%")
        else:
            st.error("❌ **Perlu Perbaikan.** Relative error > 5%")
    else:
        st.warning("⚠️ Nilai sebenarnya tidak boleh nol untuk menghitung relative error.")

# --- HALAMAN 3: AKAR PERSAMAAN ---
elif menu == "🎯 Akar Persamaan":
    st.header("🎯 Mencari Akar (Root Finding)")
    
    st.markdown("""
    **Masalah klasik:** Kapan fungsi ini menyentuh sumbu X (nilai nol)?
    
    **Aplikasi dalam teknik:**
    - Kapan arus dalam rangkaian menjadi nol?
    - Kapan tegangan drop menjadi nol?
    - Mencari titik kesetimbangan sistem
    """)
    
    # Pilih Metode
    method = st.selectbox(
        "Pilih Metode:",
        ["Bisection (Dikotomi)", "Newton-Raphson", "Secant"]
    )
    
    # Input Fungsi
    st.markdown("---")
    st.subheader("📝 Input Fungsi")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        func_input = st.text_input(
            "Masukkan fungsi f(x):",
            value="x**2 - 4",
            help="Gunakan `**` untuk pangkat. Contoh: x**2 - 4, x**3 - x - 1, sin(x) - 0.5"
        )
    
    with col2:
        st.markdown("**Contoh:**")
        st.code("x**2 - 4")
        st.code("x**3 - x - 1")
        st.code("sin(x) - 0.5")
    
    # Parse fungsi
    f = parse_expression(func_input)
    
    if f is None:
        st.warning("⚠️ Mohon perbaiki rumus fungsi sebelum melanjutkan.")
        st.info("""
        **Tips Penulisan:**
        - Gunakan `x` sebagai variabel
        - Pangkat: `x**2` bukan `x^2`
        - Fungsi trigonometri: `sin(x)`, `cos(x)`, `tan(x)`
        - Eksponensial: `exp(x)`
        - Logaritma natural: `log(x)`
        - Akar kuadrat: `sqrt(x)`
        """)
        st.stop()
    
    # Plot fungsi
    st.markdown("---")
    st.subheader("📊 Visualisasi Fungsi")
    
    try:
        x_range = st.slider("Range X untuk plot:", -20.0, 20.0, (-10.0, 10.0), 0.5)
        x = np.linspace(x_range[0], x_range[1], 500)
        y = f(x)
        
        # Pastikan y adalah array
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='f(x)', line=dict(color='blue', width=2)))
        fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="y=0")
        fig.update_layout(
            title="Grafik Fungsi f(x)",
            xaxis_title='x',
            yaxis_title='f(x)',
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Error saat plotting: {e}")
        st.info("Pastikan fungsi yang dimasukkan valid dan dapat dievaluasi untuk range x yang dipilih.")
        st.stop()
    
    # Metode Bisection
    if method == "Bisection (Dikotomi)":
        st.markdown("---")
        st.markdown("""
        <div class="concept-box">
        <h4>🎲 Konsep: Permainan Tebak Angka</h4>
        
        Bayangkan kita bermain tebak angka 1-100:
        1. Pilih tengah (50): "Terlalu besar"
        2. Pilih tengah baru (25): "Terlalu kecil"
        3. Pilih tengah lagi (37): "Terlalu besar"
        4. Dan seterusnya...
        
        **Metode Bisection** bekerja dengan cara yang sama:
        - Kita punya interval [a, b] dimana f(a) dan f(b) berlawanan tanda
        - Artinya ada akar di antara a dan b (berdasarkan Teorema Nilai Antara)
        - Kita potong tengah-tengahnya, cek tandanya, buang setengah yang salah
        - Ulangi sampai intervalnya sangat kecil
        
        **Keunggulan:** Selalu konvergen jika ada akar dalam interval
        **Kelemahan:** Lambat (mengurangi error 50% setiap iterasi)
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            a = st.number_input("Batas Bawah (a):", value=-3.0, format="%.6f")
        
        with col2:
            b = st.number_input("Batas Atas (b):", value=3.0, format="%.6f")
        
        with col3:
            tol = st.number_input("Toleransi:", value=0.0001, format="%.6f", min_value=1e-10)
        
        max_iter = st.slider("Maksimum Iterasi:", 5, 100, 20)
        
        if st.button("🚀 Hitung Akar (Bisection)", type="primary"):
            with st.spinner("Menghitung..."):
                results, msg = bisection_method(f, a, b, tol, max_iter)
            
            if results is None:
                st.error(msg)
            else:
                st.success(msg)
                
                # Tampilkan tabel
                st.subheader("📋 Tabel Iterasi")
                df = pd.DataFrame(results, columns=["Iterasi", "a", "b", "c (Akar)", "f(c)"])
                df['|b - a|'] = df['b'] - df['a']
                st.dataframe(df.style.format({
                    'a': '{:.6f}',
                    'b': '{:.6f}',
                    'c (Akar)': '{:.6f}',
                    'f(c)': '{:.6e}',
                    '|b - a|': '{:.6e}'
                }), use_container_width=True)
                
                # Hasil akhir
                final_root = results[-1][3]
                final_error = abs(results[-1][4])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Akar yang ditemukan", f"{final_root:.8f}")
                with col2:
                    st.metric("f(akar)", f"{final_error:.2e}")
                with col3:
                    st.metric("Jumlah Iterasi", len(results))
                
                # Plot hasil
                fig_result = go.Figure()
                fig_result.add_trace(go.Scatter(x=x, y=y, mode='lines', name='f(x)', line=dict(color='blue', width=2)))
                fig_result.add_hline(y=0, line_dash="dash", line_color="gray")
                fig_result.add_trace(go.Scatter(
                    x=[final_root], y=[0],
                    mode='markers',
                    marker=dict(size=15, color='red', symbol='star'),
                    name=f'Akar ≈ {final_root:.6f}'
                ))
                fig_result.update_layout(
                    title="Hasil: Akar yang Ditemukan",
                    xaxis_title='x',
                    yaxis_title='f(x)',
                    height=400
                )
                st.plotly_chart(fig_result, use_container_width=True)
    
    # Metode Newton-Raphson
    elif method == "Newton-Raphson":
        st.markdown("---")
        st.markdown("""
        <div class="concept-box">
        <h4>🎿 Konsep: Meluncur di Lereng</h4>
        
        Bayangkan Anda berdiri di lereng bukit dan ingin cepat turun ke lembah (akar):
        1. Lihat kemiringan di bawah kaki Anda (turunan)
        2. Meluncur mengikuti kemiringan itu
        3. Sampai di titik baru, lihat kemiringan lagi
        4. Ulangi sampai sampai di lembah
        
        **Metode Newton-Raphson:**
        - Mulai dari tebakan awal x₀
        - Tarik garis singgung di titik (x₀, f(x₀))
        - Garis singgung memotong sumbu X di x₁
        - x₁ biasanya lebih dekat ke akar daripada x₀
        - Rumus: x_{n+1} = x_n - f(x_n)/f'(x_n)
        
        **Keunggulan:** Sangat cepat (konvergensi kuadratik)
        **Kelemahan:** Butuh turunan, bisa gagal jika f'(x) ≈ 0
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x0 = st.number_input("Tebakan Awal (x₀):", value=3.0, format="%.6f")
        
        with col2:
            tol = st.number_input("Toleransi:", value=0.0001, format="%.6f", min_value=1e-10)
        
        with col3:
            max_iter = st.slider("Maksimum Iterasi:", 5, 50, 15)
        
        if st.button("🚀 Hitung Akar (Newton-Raphson)", type="primary"):
            with st.spinner("Menghitung..."):
                # Numerical derivative
                h = 1e-7
                def df(x_val):
                    return (f(x_val + h) - f(x_val - h)) / (2 * h)
                
                results, msg = newton_raphson_method(f, df, x0, tol, max_iter)
            
            if results is None:
                st.error(msg)
            else:
                st.success(msg)
                
                # Tampilkan tabel
                st.subheader("📋 Tabel Iterasi")
                df_results = pd.DataFrame(results, columns=["Iterasi", "xₙ", "f(xₙ)", "xₙ₊₁", "|xₙ₊₁ - xₙ|"])
                st.dataframe(df_results.style.format({
                    'xₙ': '{:.8f}',
                    'f(xₙ)': '{:.6e}',
                    'xₙ₊₁': '{:.8f}',
                    '|xₙ₊₁ - xₙ|': '{:.6e}'
                }), use_container_width=True)
                
                # Hasil akhir
                final_root = results[-1][3]
                final_error = abs(float(f(final_root)))
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Akar yang ditemukan", f"{final_root:.8f}")
                with col2:
                    st.metric("f(akar)", f"{final_error:.2e}")
                with col3:
                    st.metric("Jumlah Iterasi", len(results))
                
                # Plot hasil dengan garis singgung
                fig_result = go.Figure()
                fig_result.add_trace(go.Scatter(x=x, y=y, mode='lines', name='f(x)', line=dict(color='blue', width=2)))
                fig_result.add_hline(y=0, line_dash="dash", line_color="gray")
                
                # Tambahkan garis singgung iterasi terakhir
                if len(results) > 0:
                    x_last = results[-1][2]
                    f_last = float(f(x_last))
                    h = 1e-7
                    df_last = (float(f(x_last + h)) - float(f(x_last - h))) / (2 * h)
                    
                    # Buat garis singgung
                    x_tangent = np.linspace(x_last - 2, x_last + 2, 50)
                    y_tangent = f_last + df_last * (x_tangent - x_last)
                    
                    fig_result.add_trace(go.Scatter(
                        x=x_tangent, y=y_tangent,
                        mode='lines',
                        name='Garis Singgung',
                        line=dict(color='orange', dash='dot')
                    ))
                
                fig_result.add_trace(go.Scatter(
                    x=[final_root], y=[0],
                    mode='markers',
                    marker=dict(size=15, color='green', symbol='star'),
                    name=f'Akar ≈ {final_root:.6f}'
                ))
                fig_result.update_layout(
                    title="Hasil: Akar yang Ditemukan",
                    xaxis_title='x',
                    yaxis_title='f(x)',
                    height=400
                )
                st.plotly_chart(fig_result, use_container_width=True)
    
    # Metode Secant
    elif method == "Secant":
        st.markdown("---")
        st.markdown("""
        <div class="concept-box">
        <h4>📐 Konsep: Newton-Raphson Tanpa Turunan</h4>
        
        **Metode Secant** adalah modifikasi Newton-Raphson yang tidak memerlukan turunan:
        - Gunakan 2 titik awal: x₀ dan x₁
        - Tarik garis lurus (secant) melalui kedua titik
        - Garis secant memotong sumbu X di x₂
        - Geser: gunakan x₁ dan x₂ untuk iterasi berikutnya
        
        **Rumus:**
        x_{n+1} = x_n - f(x_n) × (x_n - x_{n-1}) / (f(x_n) - f(x_{n-1}))
        
        **Keunggulan:** Tidak butuh turunan, lebih cepat dari Bisection
        **Kelemahan:** Lebih lambat dari Newton-Raphson, bisa divergen
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x0 = st.number_input("Tebakan Awal 1 (x₀):", value=0.0, format="%.6f")
        
        with col2:
            x1 = st.number_input("Tebakan Awal 2 (x₁):", value=1.0, format="%.6f")
        
        with col3:
            tol = st.number_input("Toleransi:", value=0.0001, format="%.6f", min_value=1e-10)
        
        max_iter = st.slider("Maksimum Iterasi:", 5, 50, 15)
        
        if st.button("🚀 Hitung Akar (Secant)", type="primary"):
            with st.spinner("Menghitung..."):
                results, msg = secant_method(f, x0, x1, tol, max_iter)
            
            if results is None:
                st.error(msg)
            else:
                st.success(msg)
                
                # Tampilkan tabel
                st.subheader("📋 Tabel Iterasi")
                df_results = pd.DataFrame(results, columns=["Iterasi", "xₙ₋₁", "xₙ", "xₙ₊₁", "f(xₙ₊₁)"])
                st.dataframe(df_results.style.format({
                    'xₙ₋₁': '{:.8f}',
                    'xₙ': '{:.8f}',
                    'xₙ₊₁': '{:.8f}',
                    'f(xₙ₊₁)': '{:.6e}'
                }), use_container_width=True)
                
                # Hasil akhir
                final_root = results[-1][3]
                final_error = abs(float(f(final_root)))
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Akar yang ditemukan", f"{final_root:.8f}")
                with col2:
                    st.metric("f(akar)", f"{final_error:.2e}")
                with col3:
                    st.metric("Jumlah Iterasi", len(results))
                
                # Plot hasil
                fig_result = go.Figure()
                fig_result.add_trace(go.Scatter(x=x, y=y, mode='lines', name='f(x)', line=dict(color='blue', width=2)))
                fig_result.add_hline(y=0, line_dash="dash", line_color="gray")
                fig_result.add_trace(go.Scatter(
                    x=[final_root], y=[0],
                    mode='markers',
                    marker=dict(size=15, color='purple', symbol='star'),
                    name=f'Akar ≈ {final_root:.6f}'
                ))
                fig_result.update_layout(
                    title="Hasil: Akar yang Ditemukan",
                    xaxis_title='x',
                    yaxis_title='f(x)',
                    height=400
                )
                st.plotly_chart(fig_result, use_container_width=True)

# --- HALAMAN 4: SISTEM LINEAR ---
elif menu == "🧮 Sistem Linear":
    st.header("🧮 Sistem Persamaan Linear")
    
    st.markdown("""
    **Memecahkan masalah seperti:**
    - Berapa arus di setiap cabang rangkaian? (Hukum Kirchhoff)
    - Berapa gaya pada setiap tumpuan jembatan?
    - Bagaimana distribusi suhu dalam sistem?
    
    Semua masalah ini dapat dinyatakan dalam bentuk matriks: **Ax = b**
    """)
    
    st.markdown('<div class="concept-box">', unsafe_allow_html=True)
    st.markdown("""
    ### 🎯 Konsep Sistem Linear
    
    Sistem persamaan linear adalah kumpulan persamaan seperti:
```
    3x + 2y - z = 1
    2x - 2y + 4z = -2
    -x + 0.5y - z = 0
```
    
    Dalam bentuk matriks:
    
    **A** (Koefisien) × **x** (Unknown) = **b** (Konstanta)
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("📝 Input Sistem 3×3")
    
    st.info("💡 Masukkan koefisien untuk sistem 3 persamaan dengan 3 variabel (x₁, x₂, x₃)")
    
    # Input Matrix A
    st.markdown("**Matriks Koefisien A:**")
    col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
    
    with col1:
        st.markdown("**Persamaan 1:**")
        a11 = st.number_input("a₁₁", value=3.0, format="%.4f", key="a11")
        a12 = st.number_input("a₁₂", value=2.0, format="%.4f", key="a12")
        a13 = st.number_input("a₁₃", value=-1.0, format="%.4f", key="a13")
    
    with col2:
        st.markdown("**Persamaan 2:**")
        a21 = st.number_input("a₂₁", value=2.0, format="%.4f", key="a21")
        a22 = st.number_input("a₂₂", value=-2.0, format="%.4f", key="a22")
        a23 = st.number_input("a₂₃", value=4.0, format="%.4f", key="a23")
    
    with col3:
        st.markdown("**Persamaan 3:**")
        a31 = st.number_input("a₃₁", value=-1.0, format="%.4f", key="a31")
        a32 = st.number_input("a₃₂", value=0.5, format="%.4f", key="a32")
        a33 = st.number_input("a₃₃", value=-1.0, format="%.4f", key="a33")
    
    with col4:
        st.markdown("**Vektor b:**")
        b1 = st.number_input("b₁", value=1.0, format="%.4f", key="b1")
        b2 = st.number_input("b₂", value=-2.0, format="%.4f", key="b2")
        b3 = st.number_input("b₃", value=0.0, format="%.4f", key="b3")
    
    # Tampilkan sistem dalam bentuk LaTeX
    st.markdown("---")
    st.subheader("📐 Sistem Persamaan:")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.latex(fr"""
        \begin{{cases}}
        {a11}x_1 + {a12}x_2 + {a13}x_3 = {b1} \\
        {a21}x_1 + {a22}x_2 + {a23}x_3 = {b2} \\
        {a31}x_1 + {a32}x_2 + {a33}x_3 = {b3}
        \end{{cases}}
        """)
    
    with col2:
        st.markdown("**Bentuk Matriks:**")
        st.latex(r"""
        \begin{bmatrix} A \end{bmatrix} 
        \begin{bmatrix} x \end{bmatrix} = 
        \begin{bmatrix} b \end{bmatrix}
        """)
    
    # Tombol solve
    if st.button("🚀 Selesaikan Sistem", type="primary"):
        A = np.array([
            [a11, a12, a13],
            [a21, a22, a23],
            [a31, a32, a33]
        ])
        b = np.array([b1, b2, b3])
        
        try:
            # Hitung determinan
            det_A = np.linalg.det(A)
            
            st.markdown("---")
            st.subheader("📊 Analisis Matriks")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Determinan", f"{det_A:.6f}")
                if abs(det_A) < 1e-10:
                    st.error("⚠️ Matriks singular! (det ≈ 0)")
                    st.info("Sistem tidak memiliki solusi unik atau tidak memiliki solusi.")
                else:
                    st.success("✅ Matriks non-singular (det ≠ 0)")
            
            with col2:
                condition_number = np.linalg.cond(A)
                st.metric("Condition Number", f"{condition_number:.2f}")
                if condition_number > 1000:
                    st.warning("⚠️ Matriks ill-conditioned (sensitif terhadap error)")
                else:
                    st.success("✅ Matriks well-conditioned")
            
            # Solve
            if abs(det_A) >= 1e-10:
                sol = np.linalg.solve(A, b)
                
                st.markdown("---")
                st.subheader("✅ Solusi")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("x₁", f"{sol[0]:.6f}")
                
                with col2:
                    st.metric("x₂", f"{sol[1]:.6f}")
                
                with col3:
                    st.metric("x₃", f"{sol[2]:.6f}")
                
                # Verifikasi
                st.markdown("---")
                st.subheader("🔍 Verifikasi: A × x = b")
                
                result = A @ sol
                
                verification_df = pd.DataFrame({
                    'Persamaan': ['Persamaan 1', 'Persamaan 2', 'Persamaan 3'],
                    'A×x (Hasil)': result,
                    'b (Target)': b,
                    'Error': np.abs(result - b)
                })
                
                st.dataframe(verification_df.style.format({
                    'A×x (Hasil)': '{:.6f}',
                    'b (Target)': '{:.6f}',
                    'Error': '{:.2e}'
                }), use_container_width=True)
                
                max_error = np.max(np.abs(result - b))
                if max_error < 1e-6:
                    st.success(f"✅ Verifikasi berhasil! Max error: {max_error:.2e}")
                else:
                    st.warning(f"⚠️ Error verifikasi: {max_error:.2e}")
                
                # Visualisasi (untuk sistem 2D)
                st.markdown("---")
                st.info("""
                💡 **Metode yang Digunakan:**
                
                NumPy menggunakan algoritma LAPACK yang didasarkan pada:
                - **LU Decomposition** untuk matriks umum
                - **Gaussian Elimination** dengan partial pivoting
                - Sangat efisien dan stabil secara numerik
                """)
        
        except np.linalg.LinAlgError as e:
            st.error(f"❌ Error: Singular Matrix! {e}")
            st.info("""
            **Penyebab:**
            - Determinan = 0
            - Ada persamaan yang redundan
            - Tidak ada solusi unik
            """)

# --- HALAMAN 5: INTERPOLASI ---
elif menu == "📈 Interpolasi":
    st.header("📈 Interpolasi & Regresi")
    
    st.markdown("""
    <div class="concept-box">
    <h4>🎯 Perbedaan Interpolasi vs Regresi</h4>
    
    **Interpolasi:**
    - "Hubungkan titik-titik dengan kurva yang MELEWATI semua titik"
    - Digunakan ketika data akurat dan sedikit
    - Contoh: Menghitung suhu di jam 14:30 dari data jam 14:00 dan 15:00
    
    **Regresi:**
    - "Cari pola/tren UMUM dari data yang berantakan"
    - Digunakan ketika data banyak dan ada noise/error
    - Contoh: Memprediksi harga rumah dari data historis yang bervariasi
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    tab1, tab2 = st.tabs(["📍 Interpolasi", "📊 Regresi"])
    
    with tab1:
        st.subheader("📍 Interpolasi Polinomial")
        
        st.markdown("""
        **Prinsip:** Cari polinomial derajat n-1 yang melewati n titik data.
        
        Jika ada 4 titik data, kita cari polinomial derajat 3: P(x) = ax³ + bx² + cx + d
        """)
        
        # Input data
        col1, col2 = st.columns(2)
        
        with col1:
            x_input = st.text_area(
                "Masukkan data X (pisahkan dengan koma):",
                value="0, 1, 2, 3, 4",
                height=100
            )
        
        with col2:
            y_input = st.text_area(
                "Masukkan data Y (pisahkan dengan koma):",
                value="1, 3, 2, 5, 4",
                height=100
            )
        
        try:
            x_data = np.array([float(i.strip()) for i in x_input.split(',')])
            y_data = np.array([float(i.strip()) for i in y_input.split(',')])
            
            if len(x_data) != len(y_data):
                st.error("❌ Jumlah data X dan Y harus sama!")
            elif len(x_data) < 2:
                st.error("❌ Minimal 2 titik data diperlukan!")
            else:
                # Interpolasi
                degree = len(x_data) - 1
                poly_coef = np.polyfit(x_data, y_data, degree)
                poly = np.poly1d(poly_coef)
                
                # Generate smooth curve
                x_smooth = np.linspace(min(x_data) - 1, max(x_data) + 1, 300)
                y_smooth = poly(x_smooth)
                
                # Plot
                fig = go.Figure()
                
                # Data points
                fig.add_trace(go.Scatter(
                    x=x_data, y=y_data,
                    mode='markers',
                    name='Data Asli',
                    marker=dict(size=12, color='red', symbol='circle')
                ))
                
                # Interpolation curve
                fig.add_trace(go.Scatter(
                    x=x_smooth, y=y_smooth,
                    mode='lines',
                    name=f'Interpolasi (Derajat {degree})',
                    line=dict(color='blue', width=2)
                ))
                
                fig.update_layout(
                    title=f"Interpolasi Polinomial Derajat {degree}",
                    xaxis_title='x',
                    yaxis_title='y',
                    hovermode='x unified',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Tampilkan persamaan
                st.subheader("📐 Persamaan Polinomial")
                
                poly_str = "P(x) = "
                for i, coef in enumerate(poly_coef):
                    power = degree - i
                    if i > 0:
                        poly_str += " + " if coef >= 0 else " - "
                        coef = abs(coef)
                    
                    if power == 0:
                        poly_str += f"{coef:.4f}"
                    elif power == 1:
                        poly_str += f"{coef:.4f}x"
                    else:
                        poly_str += f"{coef:.4f}x^{power}"
                
                st.code(poly_str)
                
                # Kalkulator interpolasi
                st.markdown("---")
                st.subheader("🔢 Kalkulator Interpolasi")
                
                x_test = st.number_input(
                    "Masukkan nilai x untuk dievaluasi:",
                    value=float(np.mean(x_data)),
                    format="%.4f"
                )
                
                y_test = poly(x_test)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Input (x)", f"{x_test:.4f}")
                with col2:
                    st.metric("Output P(x)", f"{y_test:.4f}")
                
                # Cek apakah x_test dalam range data
                if x_test < min(x_data) or x_test > max(x_data):
                    st.warning("⚠️ Nilai x berada di luar range data (ekstrapolasi). Hasil mungkin tidak akurat!")
        
        except ValueError as e:
            st.error(f"❌ Error parsing data: {e}")
            st.info("Pastikan data berupa angka yang dipisahkan koma.")
    
    with tab2:
        st.subheader("📊 Regresi Linear")
        
        st.markdown("""
        **Prinsip:** Cari garis lurus y = mx + c yang "paling cocok" dengan data.
        
        Menggunakan **Least Squares Method** - meminimalkan jumlah kuadrat error.
        """)
        
        # Input data
        col1, col2 = st.columns(2)
        
        with col1:
            x_input_reg = st.text_area(
                "Masukkan data X (pisahkan dengan koma):",
                value="1, 2, 3, 4, 5, 6, 7, 8, 9, 10",
                height=100,
                key="x_reg"
            )
        
        with col2:
            y_input_reg = st.text_area(
                "Masukkan data Y (pisahkan dengan koma):",
                value="2.1, 3.9, 6.2, 8.1, 9.8, 12.2, 14.1, 15.9, 18.2, 20.1",
                height=100,
                key="y_reg"
            )
        
        try:
            x_data_reg = np.array([float(i.strip()) for i in x_input_reg.split(',')])
            y_data_reg = np.array([float(i.strip()) for i in y_input_reg.split(',')])
            
            if len(x_data_reg) != len(y_data_reg):
                st.error("❌ Jumlah data X dan Y harus sama!")
            elif len(x_data_reg) < 2:
                st.error("❌ Minimal 2 titik data diperlukan!")
            else:
                # Linear regression
                m, c = np.polyfit(x_data_reg, y_data_reg, 1)
                y_pred = m * x_data_reg + c
                
                # Calculate R²
                ss_res = np.sum((y_data_reg - y_pred) ** 2)
                ss_tot = np.sum((y_data_reg - np.mean(y_data_reg)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)
                
                # Plot
                fig = go.Figure()
                
                # Data points
                fig.add_trace(go.Scatter(
                    x=x_data_reg, y=y_data_reg,
                    mode='markers',
                    name='Data Asli',
                    marker=dict(size=10, color='red', symbol='circle')
                ))
                
                # Regression line
                x_line = np.linspace(min(x_data_reg), max(x_data_reg), 100)
                y_line = m * x_line + c
                
                fig.add_trace(go.Scatter(
                    x=x_line, y=y_line,
                    mode='lines',
                    name='Regresi Linear',
                    line=dict(color='green', width=2)
                ))
                
                # Residuals
                for i in range(len(x_data_reg)):
                    fig.add_trace(go.Scatter(
                        x=[x_data_reg[i], x_data_reg[i]],
                        y=[y_data_reg[i], y_pred[i]],
                        mode='lines',
                        line=dict(color='gray', width=1, dash='dot'),
                        showlegend=False
                    ))
                
                fig.update_layout(
                    title="Regresi Linear: y = mx + c",
                    xaxis_title='x',
                    yaxis_title='y',
                    hovermode='x unified',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Hasil
                st.markdown("---")
                st.subheader("📐 Hasil Regresi")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Slope (m)", f"{m:.4f}")
                
                with col2:
                    st.metric("Intercept (c)", f"{c:.4f}")
                
                with col3:
                    st.metric("R² (Koefisien Determinasi)", f"{r_squared:.4f}")
                
                st.code(f"y = {m:.4f}x + {c:.4f}")
                
                # Interpretasi R²
                if r_squared > 0.9:
                    st.success("✅ R² > 0.9: Model sangat baik!")
                elif r_squared > 0.7:
                    st.info("✓ R² > 0.7: Model cukup baik")
                elif r_squared > 0.5:
                    st.warning("⚠️ R² > 0.5: Model kurang baik")
                else:
                    st.error("❌ R² < 0.5: Model buruk, data mungkin tidak linear")
                
                # Prediksi
                st.markdown("---")
                st.subheader("🔮 Prediksi")
                
                x_pred = st.number_input(
                    "Masukkan nilai x untuk prediksi:",
                    value=float(np.mean(x_data_reg)),
                    format="%.4f"
                )
                
                y_pred_value = m * x_pred + c
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Input (x)", f"{x_pred:.4f}")
                with col2:
                    st.metric("Prediksi (y)", f"{y_pred_value:.4f}")
        
        except ValueError as e:
            st.error(f"❌ Error parsing data: {e}")
            st.info("Pastikan data berupa angka yang dipisahkan koma.")

# --- HALAMAN 6: INTEGRAL & PDB ---
elif menu == "⚙️ Integral & PDB":
    st.header("⚙️ Integral & Persamaan Diferensial")
    
    tab1, tab2 = st.tabs(["📐 Integral Numerik", "🔌 Simulasi Rangkaian RC"])
    
    with tab1:
        st.subheader("📐 Integral Numerik")
        
        st.markdown("""
        <div class="concept-box">
        <h4>🎯 Konsep Feynman: "Jumlahan Kecil-Kecil"</h4>
        
        Bayangkan kita ingin tahu luas area di bawah kurva:
        
        1. Potong area menjadi banyak trapesium kecil
        2. Hitung luas masing-masing trapesium: (tinggi_kiri + tinggi_kanan) / 2 × lebar
        3. Jumlahkan semua luas trapesium
        4. Semakin banyak trapesium → hasil semakin akurat
        
        **Rumus Trapesium:**
        ∫ f(x)dx ≈ (Δx/2) × [f(x₀) + 2f(x₁) + 2f(x₂) + ... + 2f(xₙ₋₁) + f(xₙ)]
        </div>
        """, unsafe_allow_html=True)
        
        # Input fungsi
        col1, col2 = st.columns([3, 1])
        
        with col1:
            func_input_int = st.text_input(
                "Masukkan fungsi f(x):",
                value="sin(x) + 1",
                key="func_integral",
                help="Contoh: sin(x), x**2, exp(-x**2)"
            )
        
        with col2:
            st.markdown("**Contoh:**")
            st.code("sin(x)")
            st.code("x**2")
            st.code("exp(-x)")
        
        f_int = parse_expression(func_input_int)
        
        if f_int is None:
            st.warning("⚠️ Mohon perbaiki rumus fungsi.")
            st.stop()
        
        # Parameter integral
        col1, col2, col3 = st.columns(3)
        
        with col1:
            a_int = st.number_input("Batas Bawah (a):", value=0.0, format="%.4f")
        
        with col2:
            b_int = st.number_input("Batas Atas (b):", value=np.pi, format="%.4f")
        
        with col3:
            n_int = st.slider("Jumlah Segmen (n):", 4, 100, 10)
        
        if st.button("🚀 Hitung Integral", type="primary"):
            try:
                # Generate points
                x_points = np.linspace(a_int, b_int, n_int + 1)
                y_points = f_int(x_points)
                
                # Pastikan y_points adalah array
                if not isinstance(y_points, np.ndarray):
                    y_points = np.array(y_points)
                
                # Trapezoidal rule
                dx = (b_int - a_int) / n_int
                integral_result = 0.5 * dx * (y_points[0] + 2 * np.sum(y_points[1:-1]) + y_points[-1])
                
                # Plot
                fig = go.Figure()
                
                # Function curve
                x_smooth = np.linspace(a_int, b_int, 300)
                y_smooth = f_int(x_smooth)
                
                # Pastikan y_smooth adalah array
                if not isinstance(y_smooth, np.ndarray):
                    y_smooth = np.array(y_smooth)
                
                fig.add_trace(go.Scatter(
                    x=x_smooth, y=y_smooth,
                    mode='lines',
                    name='f(x)',
                    line=dict(color='blue', width=3)
                ))
                
                # Trapezoids
                for i in range(n_int):
                    fig.add_trace(go.Scatter(
                        x=[x_points[i], x_points[i], x_points[i+1], x_points[i+1], x_points[i]],
                        y=[0, float(y_points[i]), float(y_points[i+1]), 0, 0],
                        fill='toself',
                        fillcolor='rgba(0, 100, 200, 0.2)',
                        line=dict(color='rgba(0, 100, 200, 0.5)', width=1),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                
                fig.update_layout(
                    title=f"Metode Trapesium dengan {n_int} Segmen",
                    xaxis_title='x',
                    yaxis_title='f(x)',
                    hovermode='x unified',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Hasil
                st.markdown("---")
                st.subheader("📊 Hasil Perhitungan")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Nilai Integral", f"{integral_result:.6f}")
                
                with col2:
                    st.metric("Jumlah Segmen", n_int)
                
                with col3:
                    st.metric("Lebar Segmen (Δx)", f"{dx:.6f}")
                
                # Info tambahan
                st.info("""
                💡 **Tips:**
                - Semakin banyak segmen → hasil semakin akurat
                - Metode Trapesium memiliki error O(h²)
                - Untuk fungsi yang sangat non-linear, gunakan lebih banyak segmen
                """)
                
                # Perbandingan dengan metode lain (jika memungkinkan)
                try:
                    from scipy import integrate
                    
                    # Wrapper untuk scipy.integrate.quad yang hanya menerima scalar
                    def f_scalar(x_val):
                        result = f_int(x_val)
                        if isinstance(result, np.ndarray):
                            return float(result[0] if len(result) > 0 else result)
                        return float(result)
                    
                    result_scipy, error = integrate.quad(f_scalar, a_int, b_int)
                    
                    st.markdown("---")
                    st.subheader("🔬 Verifikasi dengan SciPy")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Hasil Trapesium", f"{integral_result:.8f}")
                    
                    with col2:
                        st.metric("Hasil SciPy (Referensi)", f"{result_scipy:.8f}")
                    
                    with col3:
                        error_diff = abs(integral_result - result_scipy)
                        st.metric("Selisih", f"{error_diff:.2e}")
                    
                    if error_diff < 0.001:
                        st.success("✅ Hasil sangat akurat!")
                    elif error_diff < 0.01:
                        st.info("✓ Hasil cukup akurat")
                    else:
                        st.warning("⚠️ Pertimbangkan menambah jumlah segmen untuk akurasi lebih baik")
                
                except Exception as e:
                    st.warning(f"Tidak dapat melakukan verifikasi dengan SciPy: {e}")
            
            except Exception as e:
                st.error(f"❌ Error: {e}")
    
    with tab2:
        st.subheader("🔌 Simulasi Rangkaian RC (PDB)")
        
        st.markdown("""
        <div class="concept-box">
        <h4>⚡ Studi Kasus: Charging Capacitor</h4>
        
        **Rangkaian RC** terdiri dari resistor (R) dan kapasitor (C) yang dihubungkan seri.
        
        **Persamaan Diferensial:**
        dVc/dt = (Vin - Vc) / (R×C)
        
        **Dimana:**
        - Vc = Tegangan kapasitor (V)
        - Vin = Tegangan input (V)
        - R = Resistansi (Ω)
        - C = Kapasitansi (F)
        - τ = R×C = Time constant (detik)
        
        **Interpretasi:**
        - τ = waktu yang dibutuhkan kapasitor untuk charge ~63% dari nilai akhir
        - Setelah 5τ, kapasitor sudah ~99% terisi penuh
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("⚙️ Parameter Rangkaian")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            R = st.slider("Resistansi R (kΩ):", 1.0, 50.0, 10.0, 0.5) * 1000  # Convert to Ω
        
        with col2:
            C = st.slider("Kapasitansi C (µF):", 10.0, 1000.0, 100.0, 10.0) * 1e-6  # Convert to F
        
        with col3:
            Vin = st.number_input("Tegangan Input Vin (V):", value=5.0, min_value=0.1, format="%.2f")
        
        # Hitung time constant
        tau = R * C
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Time Constant (τ)", f"{tau:.4f} s")
        
        with col2:
            st.metric("Waktu untuk 99% charge (≈5τ)", f"{5*tau:.4f} s")
        
        # Parameter simulasi
        st.markdown("---")
        st.subheader("🎮 Parameter Simulasi")
        
        col1, col2 = st.columns(2)
        
        with col1:
            t_max_factor = st.slider("Durasi simulasi (dalam satuan τ):", 1.0, 10.0, 5.0, 0.5)
            t_max = t_max_factor * tau
        
        with col2:
            h = st.select_slider(
                "Time step (h):",
                options=[0.001, 0.005, 0.01, 0.05, 0.1],
                value=0.01
            )
        
        if st.button("🚀 Jalankan Simulasi", type="primary"):
            with st.spinner("Mensimulasikan..."):
                # Euler Method
                steps = int(t_max / h)
                t_vals = np.zeros(steps + 1)
                vc_vals = np.zeros(steps + 1)
                
                vc = 0  # Initial condition
                t = 0
                
                for i in range(steps):
                    t_vals[i] = t
                    vc_vals[i] = vc
                    
                    # Euler update
                    dvc_dt = (Vin - vc) / tau
                    vc = vc + dvc_dt * h
                    t = t + h
                
                t_vals[-1] = t
                vc_vals[-1] = vc
                
                # Solusi analitik untuk perbandingan
                t_analytical = np.linspace(0, t_max, 1000)
                vc_analytical = Vin * (1 - np.exp(-t_analytical / tau))
                
                # Plot
                fig = go.Figure()
                
                # Simulasi Euler
                fig.add_trace(go.Scatter(
                    x=t_vals, y=vc_vals,
                    mode='lines',
                    name='Euler Method (Numerik)',
                    line=dict(color='blue', width=2)
                ))
                
                # Solusi analitik
                fig.add_trace(go.Scatter(
                    x=t_analytical, y=vc_analytical,
                    mode='lines',
                    name='Solusi Analitik',
                    line=dict(color='red', width=2, dash='dash')
                ))
                
                # Reference lines
                fig.add_hline(
                    y=Vin,
                    line_dash="dot",
                    line_color="green",
                    annotation_text=f"Vin = {Vin}V"
                )
                
                fig.add_hline(
                    y=0.63 * Vin,
                    line_dash="dot",
                    line_color="orange",
                    annotation_text="63% Vin"
                )
                
                fig.add_vline(
                    x=tau,
                    line_dash="dot",
                    line_color="purple",
                    annotation_text="τ"
                )
                
                fig.update_layout(
                    title="Respon Tegangan Kapasitor vs Waktu",
                    xaxis_title='Waktu (s)',
                    yaxis_title='Tegangan Kapasitor Vc (V)',
                    hovermode='x unified',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Analisis hasil
                st.markdown("---")
                st.subheader("📊 Analisis Hasil")
                
                # Hitung error
                vc_analytical_at_points = Vin * (1 - np.exp(-t_vals / tau))
                max_error = np.max(np.abs(vc_vals - vc_analytical_at_points))
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Vc akhir (Simulasi)", f"{vc_vals[-1]:.4f} V")
                
                with col2:
                    st.metric("Vc akhir (Teori)", f"{Vin:.4f} V")
                
                with col3:
                    st.metric("Max Error", f"{max_error:.4f} V")
                
                # Milestones
                st.markdown("---")
                st.subheader("🎯 Milestone Charging")
                
                milestones = [0.63, 0.86, 0.95, 0.98, 0.99]
                milestone_data = []
                
                for percent in milestones:
                    target_voltage = percent * Vin
                    # Find time when this voltage is reached
                    idx = np.argmax(vc_vals >= target_voltage)
                    if idx > 0:
                        time_reached = t_vals[idx]
                        milestone_data.append({
                            'Persentase': f"{percent*100:.0f}%",
                            'Tegangan Target': f"{target_voltage:.3f} V",
                            'Waktu Tercapai': f"{time_reached:.4f} s",
                            'Dalam satuan τ': f"{time_reached/tau:.2f}τ"
                        })
                
                df_milestones = pd.DataFrame(milestone_data)
                st.dataframe(df_milestones, use_container_width=True)
                
                st.info("""
                💡 **Interpretasi:**
                - Pada t = τ: kapasitor terisi ~63%
                - Pada t = 2τ: kapasitor terisi ~86%
                - Pada t = 3τ: kapasitor terisi ~95%
                - Pada t = 5τ: kapasitor terisi ~99% (praktis penuh)
                
                **Metode Euler** adalah metode numerik paling sederhana untuk menyelesaikan PDB.
                Error berkurang dengan memperkecil time step (h).
                """)

# --- FOOTER ---
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📚 Tentang Aplikasi

**Metode Numerik Companion**

Aplikasi pembelajaran interaktif untuk mata kuliah Metode Numerik.

**Fitur:**
- ✅ Analisis Galat
- ✅ Akar Persamaan (3 metode)
- ✅ Sistem Linear
- ✅ Interpolasi & Regresi
- ✅ Integral & PDB

---

**Dibuat untuk:**
S1 Teknik Elektro - ELT60214

© 2025 Metode Numerik
""")

# Tips of the day
tips = [
    "💡 Tip: Gunakan toleransi 1e-6 untuk akurasi yang baik tanpa komputasi berlebihan.",
    "💡 Tip: Newton-Raphson lebih cepat dari Bisection, tapi butuh tebakan awal yang baik.",
    "💡 Tip: Interpolasi bagus untuk data sedikit dan akurat, Regresi untuk data banyak dengan noise.",
    "💡 Tip: Time constant τ = R×C menentukan seberapa cepat kapasitor terisi.",
    "💡 Tip: Semakin kecil time step pada simulasi, semakin akurat hasilnya (tapi lebih lambat)."
]

import random
st.sidebar.info(random.choice(tips))
