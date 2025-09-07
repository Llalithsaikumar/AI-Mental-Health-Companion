import { createTheme } from '@mui/material/styles';

// Color palette - Classic minimalist colors
const colors = {
  primary: {
    main: '#1a1a1a',
    light: '#4a4a4a',
    dark: '#000000',
    contrastText: '#FFFFFF'
  },
  secondary: {
    main: '#707070',
    light: '#9e9e9e',
    dark: '#505050',
    contrastText: '#FFFFFF'
  },
  success: {
    main: '#4c6a4c',
    light: '#6b8c6b',
    dark: '#2d4a2d',
  },
  error: {
    main: '#a04040',
    light: '#c26666',
    dark: '#802121',
  },
  warning: {
    main: '#96782e',
    light: '#b9985a',
    dark: '#745a16',
  },
  info: {
    main: '#3b5965',
    light: '#607c88',
    dark: '#243a44',
  },
  background: {
    default: '#f5f5f5',
    paper: '#FFFFFF',
    dark: '#eeeeee'
  },
  text: {
    primary: '#262626',
    secondary: '#595959',
    disabled: '#999999',
  },
};

// Create a theme instance
const theme = createTheme({
  palette: {
    primary: colors.primary,
    secondary: colors.secondary,
    success: colors.success,
    error: colors.error,
    warning: colors.warning,
    info: colors.info,
    background: colors.background,
    text: colors.text,
  },
  typography: {
    fontFamily: "'Georgia', 'Times New Roman', 'Garamond', serif",
    h1: {
      fontWeight: 500,
      fontSize: '2.3rem',
      lineHeight: 1.2,
      letterSpacing: '-0.01em',
    },
    h2: {
      fontWeight: 500,
      fontSize: '1.8rem',
      lineHeight: 1.3,
      letterSpacing: '-0.01em',
    },
    h3: {
      fontWeight: 500,
      fontSize: '1.6rem',
      lineHeight: 1.4,
    },
    h4: {
      fontWeight: 500,
      fontSize: '1.4rem',
      lineHeight: 1.4,
    },
    h5: {
      fontWeight: 500,
      fontSize: '1.2rem',
      lineHeight: 1.5,
    },
    h6: {
      fontWeight: 500,
      fontSize: '1rem',
      lineHeight: 1.6,
    },
    body1: {
      fontFamily: "'Helvetica', 'Arial', sans-serif",
      fontSize: '1rem',
      lineHeight: 1.6,
      letterSpacing: '0.01em',
    },
    body2: {
      fontFamily: "'Helvetica', 'Arial', sans-serif",
      fontSize: '0.875rem',
      lineHeight: 1.6,
    },
    button: {
      textTransform: 'none',
      fontWeight: 500,
    },
  },
  shape: {
    borderRadius: 12,
  },
  shadows: [
    'none',
    '0px 2px 4px rgba(0, 0, 0, 0.05)',
    '0px 4px 8px rgba(0, 0, 0, 0.05)',
    '0px 8px 16px rgba(0, 0, 0, 0.05)',
    '0px 16px 24px rgba(0, 0, 0, 0.05)',
    '0px 24px 32px rgba(0, 0, 0, 0.05)',
    '0px 32px 40px rgba(0, 0, 0, 0.05)',
    '0px 40px 48px rgba(0, 0, 0, 0.05)',
    // ... other shadow levels
    '0px 24px 48px rgba(0, 0, 0, 0.2)',
  ],
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          padding: '10px 24px',
          boxShadow: '0px 4px 8px rgba(0, 0, 0, 0.1)',
          transition: 'all 0.3s ease',
          '&:hover': {
            transform: 'translateY(-2px)',
            boxShadow: '0px 6px 12px rgba(0, 0, 0, 0.15)',
          },
        },
        containedPrimary: {
          background: `linear-gradient(45deg, ${colors.primary.main} 30%, ${colors.primary.light} 90%)`,
        },
        containedSecondary: {
          background: `linear-gradient(45deg, ${colors.secondary.main} 30%, ${colors.secondary.light} 90%)`,
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          boxShadow: '0px 4px 20px rgba(0, 0, 0, 0.08)',
          borderRadius: 16,
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          borderRadius: 16,
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          marginBottom: 16,
        },
      },
    },
  },
});

export default theme;
